import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import uuid
import json
from datetime import datetime, timezone
from urllib import request as urllib_request, error as urllib_error

from dotenv import load_dotenv
load_dotenv()  # Load .env file for environment variables

from flask import Flask, request, render_template, redirect, url_for, session, send_from_directory, abort
from flask import jsonify
from tensorflow import keras
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.cm as cm # <--- Added for Grad-CAM colors
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
try:
    import firebase_admin
    from firebase_admin import credentials, auth, firestore
except ImportError:
    firebase_admin = None
    credentials = None
    auth = None
    firestore = None

app = Flask(__name__)


# Set upload folder to absolute path relative to app.py
base_dir = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(base_dir, 'static', 'uploads')

app.secret_key = os.environ.get("FLASK_SECRET_KEY", "Tamim2731")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
FIREBASE_WEB_API_KEY = os.environ.get("FIREBASE_WEB_API_KEY", "").strip()

# Firebase Configuration from user
FIREBASE_CONFIG = {
    "apiKey": "AIzaSyC2nOO33n2tRg5LXN9bbdZXnHGxlaGa1Hk",
    "authDomain": "dr-assistance-b23de.firebaseapp.com",
    "projectId": "dr-assistance-b23de",
    "storageBucket": "dr-assistance-b23de.firebasestorage.app",
    "messagingSenderId": "960645890918",
    "appId": "1:960645890918:web:0c491d9ae42b20ffda3ea7",
    "measurementId": "G-EC78K2BM56"
}

# Simple in-memory user store for demo purposes.
users = {
    "demo@example.com": {
        "name": "Demo Doctor",
        "bmdc": "BMDC-00001",
        "password": generate_password_hash("123456")
    }
}
firebase_db = None


def initialize_firebase():
    global firebase_db
    if firebase_admin is None:
        return

    # এখানে সরাসরি আপনার ফাইলের নাম 'firebase-key.json' নির্দিষ্ট করে দেওয়া হয়েছে
    cred_name = "firebase-key.json"
    
    # বর্তমান app.py ফাইলের লোকেশন অনুযায়ী সঠিক ও পূর্ণাঙ্গ পাথ তৈরি করা
    cred_path = os.path.join(base_dir, cred_name)

    # চেক করা হচ্ছে ফাইলটি ওই পাথে আছে কি না
    if not os.path.exists(cred_path):
        print(f"❌ Firebase key not found at: {cred_path}")
        print("💡 অনুগ্রহ করে নিশ্চিত করুন যে 'firebase-key.json' ফাইলটি আপনার app.py এর পাশেই আছে।")
        return

    if not firebase_admin._apps:
        try:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            print("✅ Firebase successfully connected!")
        except Exception as e:
            print(f"❌ Firebase initialization error: {e}")
            return

    firebase_db = firestore.client()

# ফাংশনটি কল করা
initialize_firebase()


# Get the absolute path to the model file relative to this app.py file
MODEL_PATH = os.path.join(base_dir, "Proposed_model.h5")
if not os.path.exists(MODEL_PATH):
    # Backward compatibility - try lowercase version
    MODEL_PATH = os.path.join(base_dir, "proposed_model.h5")
model = keras.models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
class_names = ['colon_aca', 'colon_bnt', 'lung_aca', 'lung_bnt', 'lung_scc']

# Preprocess function
def preprocess_image(image_path):
    img = Image.open(image_path).resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def firebase_enabled():
    return firebase_db is not None and auth is not None


def firebase_email_login(email, password):
    if not FIREBASE_WEB_API_KEY:
        return None, "Firebase Web API key is missing. Set FIREBASE_WEB_API_KEY."

    login_url = (
        f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"
    )
    payload = json.dumps({
        "email": email,
        "password": password,
        "returnSecureToken": True
    }).encode("utf-8")
    req = urllib_request.Request(
        login_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib_request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data, None
    except urllib_error.HTTPError as e:
        try:
            err_data = json.loads(e.read().decode("utf-8"))
            msg = err_data.get("error", {}).get("message", "Login failed.")
        except Exception:
            msg = "Login failed."
        return None, msg
    except Exception:
        return None, "Firebase authentication request failed."


def get_next_token_number():
    if firebase_db is None or firestore is None:
        return f"T-{int(datetime.now(tz=timezone.utc).timestamp())}"

    token_ref = firebase_db.collection("counters").document("patient_token")

    @firestore.transactional
    def update_counter(transaction, ref):
        snapshot = ref.get(transaction=transaction)
        last_number = 0
        if snapshot.exists:
            data = snapshot.to_dict() or {}
            last_number = int(data.get("last_number", 0))
        next_number = last_number + 1
        transaction.set(
            ref,
            {"last_number": next_number, "updated_at": firestore.SERVER_TIMESTAMP},
            merge=True
        )
        return next_number
    transaction = firebase_db.transaction()
    next_number = update_counter(transaction, token_ref)
    return f"T-{next_number:05d}"


def find_last_conv_layer_name(keras_model):
    for layer in reversed(keras_model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    return None


def generate_gradcam_overlay(image_path, class_index):
    if model is None:
        return None

    last_conv_layer_name = find_last_conv_layer_name(model)
    if not last_conv_layer_name:
        return None

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    base_img = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(base_img, axis=0)

    grad_model = keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-12)
    heatmap = heatmap.numpy()

    heatmap_uint8 = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]

    jet_heatmap_img = Image.fromarray(np.uint8(jet_heatmap * 255)).resize((224, 224), Image.BILINEAR)
    jet_heatmap_array = np.array(jet_heatmap_img).astype("float32") / 255.0

    blended = (base_img * 0.4) + (jet_heatmap_array * 0.6)
    blended_uint8 = np.uint8(np.clip(blended, 0, 1) * 255)

    gradcam_filename = f"gradcam_{os.path.basename(image_path)}"
    gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
    Image.fromarray(blended_uint8).save(gradcam_path)
    
    return gradcam_filename

def render_page(template_name, **context):
    return render_template(template_name, user_email=session.get("user_email"), **context)


@app.route("/", methods=["GET"])
def index():
    if session.get("user_email"):
        return redirect(url_for("doctor_page"))
    return redirect(url_for("home_page"))


@app.route("/home", methods=["GET"])
def home_page():
    return render_page("home.html", active_page="home")


@app.route("/about", methods=["GET"])
def about_page():
    return render_page("about.html", active_page="about")


@app.route("/doctor", methods=["GET"])
def doctor_page():
    if not session.get("user_email"):
        return render_page(
            "login.html",
            active_page="login",
            login_error="Please login first to access Doctor page."
        )
    last_token = (session.get("last_report") or {}).get("token_number")
    return render_page(
        "doctor.html",
        active_page="doctor",
        doctor_nav=True,
        doctor_name=session.get("doctor_name", session.get("user_email", "Doctor")),
        last_token_number=last_token
    )


@app.route("/report", methods=["GET"])
def report_page():
    if not session.get("user_email"):
        return render_page(
            "login.html",
            active_page="login",
            login_error="Please login first to access Report page."
        )
    report_data = session.get("last_report")
    if not report_data:
        return render_page(
            "report.html",
            active_page="report",
            report_error="No report yet. Please submit a check from Doctor page first."
        )
    return render_page("report.html", active_page="report", **report_data)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_page("login.html", active_page="login")

    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")

    if not email or not password:
        return render_page(
            "login.html",
            active_page="login",
            login_error="Please enter both email and password."
        )

    login_ok = False
    doctor_name = None
    firebase_uid = None

    if firebase_enabled():
        login_data, login_error = firebase_email_login(email, password)
        if login_data:
            login_ok = True
            firebase_uid = login_data.get("localId")
            user_doc = firebase_db.collection("users").document(firebase_uid).get()
            if user_doc.exists:
                doctor_name = (user_doc.to_dict() or {}).get("name")
        else:
            return render_page(
                "login.html",
                active_page="login",
                login_error=f"Invalid email or password. ({login_error})"
            )
    else:
        user_record = users.get(email)
        if user_record and check_password_hash(user_record.get("password", ""), password):
            login_ok = True
            doctor_name = user_record.get("name")
        else:
            return render_page(
                "login.html",
                active_page="login",
                login_error="Invalid email or password."
            )

    if not login_ok:
        return render_page(
            "login.html",
            active_page="login",
            login_error="Invalid email or password."
        )

    session["user_email"] = email
    session["doctor_name"] = doctor_name or email
    session["firebase_uid"] = firebase_uid
    return render_page(
        "doctor.html",
        active_page="doctor",
        doctor_nav=True,
        doctor_name=session["doctor_name"],
        login_success="Login successful. You can now check final results on Doctor page."
    )


@app.route("/login/google", methods=["GET"])
def login_google():
    return redirect(url_for("login_page"))


@app.route("/firebase/google/login", methods=["POST"])
def firebase_google_login():
    try:
        data = request.get_json()
        id_token = data.get("idToken")
        
        if not id_token:
            return {"success": False, "error": "No ID token provided"}, 400
        
        if firebase_admin and auth:
            try:
                decoded_token = auth.verify_id_token(id_token)
                email = decoded_token.get("email")
                name = decoded_token.get("name", email.split("@")[0])
                
                session["user_email"] = email
                session["doctor_name"] = name
                session["firebase_uid"] = decoded_token.get("uid")
                session["oauth_provider"] = "google"
                
                return {"success": True, "redirect": url_for("doctor_page")}
            except Exception as e:
                return {"success": False, "error": f"Token verification failed: {str(e)}"}, 401
        
        import base64
        parts = id_token.split(".")
        if len(parts) >= 2:
            payload = parts[1]
            padding = 4 - len(payload) % 4
            if padding != 4:
                payload += "=" * padding
            user_info = json.loads(base64.b64decode(payload))
            email = user_info.get("email", "google_user@example.com")
            name = user_info.get("name", "Google User")
            
            session["user_email"] = email
            session["doctor_name"] = name
            session["oauth_provider"] = "google"
            
            return {"success": True, "redirect": url_for("doctor_page")}
        
        return {"success": False, "error": "Invalid token format"}, 400
        
    except Exception as e:
        return {"success": False, "error": str(e)}, 500


@app.route("/oauth/callback", methods=["GET", "POST"])
def oauth_callback():
    error = request.args.get("error")
    if error:
        return render_page(
            "login.html",
            active_page="login",
            login_error=f"Google login failed: {error}"
        )
    
    code = request.args.get("code")
    state = request.args.get("state")
    
    if state != session.get("oauth_state"):
        return render_page(
            "login.html",
            active_page="login",
            login_error="Invalid OAuth state. Please try again."
        )
    
    if not code:
        return render_page(
            "login.html",
            active_page="login",
            login_error="No authorization code received."
        )
    
    if not FIREBASE_WEB_API_KEY:
        return render_page(
            "login.html",
            active_page="login",
            login_error="Firebase Web API key not configured."
        )
    
    token_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithIdp?key={FIREBASE_WEB_API_KEY}"
    
    post_data = {
        "postBody": f"code={code}&grant_type=authorization_code&client_id=960645890918.apps.googleusercontent.com",
        "requestUri": f"{request.host_url}oauth/callback",
        "returnIdpCredentials": True,
        "returnSecureToken": True
    }
    
    try:
        req = urllib_request.Request(
            token_url,
            data=json.dumps(post_data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib_request.urlopen(req, timeout=15) as response:
            token_data = json.loads(response.read().decode("utf-8"))
            
            id_token = token_data.get("idToken")
            if id_token:
                import base64
                parts = id_token.split(".")
                if len(parts) >= 2:
                    payload = parts[1]
                    padding = 4 - len(payload) % 4
                    if padding != 4:
                        payload += "=" * padding
                    user_info = json.loads(base64.b64decode(payload))
                    email = user_info.get("email", "google_user@example.com")
                    name = user_info.get("name", "Google User")
                else:
                    email = "google_user@example.com"
                    name = "Google User"
                
                session["user_email"] = email
                session["doctor_name"] = name
                session["oauth_provider"] = "google"
                session.pop("oauth_state", None)
                
                return render_page(
                    "doctor.html",
                    active_page="doctor",
                    doctor_nav=True,
                    doctor_name=session["doctor_name"],
                    login_success="Successfully signed in with Google!"
                )
    except Exception as e:
        return render_page(
            "login.html",
            active_page="login",
            login_error=f"Google authentication failed: {str(e)}"
        )
    
    return render_page(
        "login.html",
        active_page="login",
        login_error="Failed to complete Google login."
    )
    
    
@app.route('/history')
def get_history():
    user_email = session.get('user_email') 
    if not user_email:
        return redirect('/login')

    docs = firebase_db.collection('reports').where('doctor_email', '==', user_email).stream()
    
    history_data = []
    for doc in docs:
        data = doc.to_dict()
        data['id'] = doc.id
        
        if 'created_at' in data and data['created_at']:
            try:
                data['formatted_date'] = data['created_at'].strftime('%Y-%m-%d %I:%M %p')
            except:
                data['formatted_date'] = str(data['created_at'])
        else:
            data['formatted_date'] = 'N/A'
            
        history_data.append(data)

    history_data = sorted(history_data, key=lambda x: x.get('created_at', 0), reverse=True)
    return render_template('history.html', history=history_data)


@app.route('/view_report/<report_id>')
def view_report(report_id):
    user_email = session.get('user_email')
    if not user_email:
        return redirect('/login')

    doc_ref = firebase_db.collection('reports').document(report_id)
    doc = doc_ref.get()

    if doc.exists:
        report_data = doc.to_dict()
        if report_data.get('doctor_email') != user_email:
            return "Unauthorized access!", 403

        return render_template("report.html", 
                               active_page="report", 
                               **report_data)
    else:
        return "Report not found!", 404
    
    

@app.route('/delete_history/<history_id>', methods=['POST'])
def delete_history(history_id):
    user_email = session.get('user_email')
    if user_email:
        firebase_db.collection('reports').document(history_id).delete()
        return jsonify({"status": "success"})
    return jsonify({"status": "error"})


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_page("register.html", active_page="register")

    name = request.form.get("name", "").strip()
    bmdc = request.form.get("bmdc", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")

    if not name or not bmdc or not email or not password or not confirm_password:
        return render_page(
            "register.html",
            active_page="register",
            register_error="Please fill all fields."
        )

    if password != confirm_password:
        return render_page(
            "register.html",
            active_page="register",
            register_error="Passwords do not match."
        )

    if firebase_enabled():
        try:
            created_user = auth.create_user(
                email=email,
                password=password,
                display_name=name
            )
            firebase_db.collection("users").document(created_user.uid).set({
                "name": name,
                "bmdc": bmdc,
                "email": email,
                "created_at": firestore.SERVER_TIMESTAMP
            }, merge=True)
            
            return render_page(
                "login.html",
                active_page="login",
                register_success="Registration complete. Please login."
            )
        except Exception as e:
            return render_page(
                "register.html",
                active_page="register",
                register_error=f"Registration failed: {str(e)}"
            )
    else:
        return render_page(
            "register.html",
            active_page="register",
            register_error="System Error: Firebase is not connected! Please check your firebase-key.json file."
        )


@app.route("/logout", methods=["GET"])
def logout():
    session.pop("user_email", None)
    session.pop("doctor_name", None)
    session.pop("firebase_uid", None)
    session.pop("last_report", None)
    return redirect(url_for("home_page"))


@app.route("/predict", methods=["POST"])
def predict():
    user_email = session.get("user_email")
    if not user_email:
        return render_page(
            "login.html",
            active_page="login",
            login_error="Please login first to upload and check results."
        )

    if model is None:
        return render_page(
            "doctor.html",
            active_page="doctor",
            doctor_nav=True,
            predict_error="Model file not found. Please add proposed_model.h5 or Proposed_model.h5."
        )

    if "file" not in request.files:
        return render_page(
            "doctor.html",
            active_page="doctor",
            doctor_nav=True,
            predict_error="No file uploaded."
        )

    file = request.files["file"]
    if file.filename == "":
        return render_page(
            "doctor.html",
            active_page="doctor",
            doctor_nav=True,
            predict_error="Please choose an image file."
        )

    if not allowed_file(file.filename):
        return render_page(
            "doctor.html",
            active_page="doctor",
            doctor_nav=True,
            predict_error="Unsupported file type. Please upload png/jpg/jpeg/bmp/webp."
        )

    patient_name = request.form.get("patient_name", "").strip()
    doctor_name = request.form.get("doctor_name", "").strip() or session.get("doctor_name", "")
    if not patient_name:
        return render_page(
            "doctor.html",
            active_page="doctor",
            doctor_nav=True,
            doctor_name=doctor_name,
            predict_error="Please enter patient name."
        )

    cancer_type = request.form.get("cancer_type", "").strip().lower()
    if cancer_type not in ["colon", "lung"]:
        return render_page(
            "doctor.html",
            active_page="doctor",
            doctor_nav=True,
            doctor_name=doctor_name,
            predict_error="Please select Colon Cancer or Lung Cancer before prediction."
        )

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    original_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)

    try:
        with Image.open(filepath) as verify_img:
            verify_img.verify()
    except Exception:
        os.remove(filepath)
        return render_page(
            "doctor.html",
            active_page="doctor",
            doctor_nav=True,
            predict_error="Invalid image file. Please upload a valid pathology image."
        )

    try:
        img_array = preprocess_image(filepath)
        preds = model.predict(img_array)
        class_index = np.argmax(preds[0])
        confidence = float(preds[0][class_index])
        predicted_class = class_names[class_index]
        gradcam_filename = generate_gradcam_overlay(filepath, class_index)
    except Exception:
        return render_page(
            "doctor.html",
            active_page="doctor",
            doctor_nav=True,
            predict_error="Prediction failed. Please try another image."
        )

    predicted_cancer_type = "colon" if predicted_class.startswith("colon_") else "lung"

    if predicted_cancer_type == cancer_type:
        final_result = f"Final Output: {predicted_class} (matches selected {cancer_type} cancer check)"
    else:
        final_result = (
            f"Final Output: {predicted_class} (model indicates {predicted_cancer_type} class, "
            f"but you selected {cancer_type})"
        )

    token_number = get_next_token_number()
    report_id = uuid.uuid4().hex
    report_payload = {
        "report_id": report_id,
        "filename": unique_name,
        "pred_class": predicted_class,
        "confidence": f"{confidence:.4f}",
        "final_result": final_result,
        "selected_cancer_type": cancer_type,
        "gradcam_filename": gradcam_filename,
        "patient_name": patient_name,
        "doctor_name": doctor_name,
        "token_number": token_number
    }

    if firebase_db is not None:
        firebase_db.collection("reports").document(report_id).set({
            "report_id": report_id,
            "doctor_email": user_email,
            "doctor_name": doctor_name,
            "patient_name": patient_name,
            "token_number": token_number,
            "selected_cancer_type": cancer_type,
            "pred_class": predicted_class,
            "confidence": confidence,
            "final_result": final_result,
            "filename": unique_name,
            "gradcam_filename": gradcam_filename,
            "created_at": firestore.SERVER_TIMESTAMP
        }, merge=True)

    session["last_report"] = report_payload
    return redirect(url_for("report_page"))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        abort(404)


@app.route("/pic/<path:filename>")
def pic_file(filename):
    return send_from_directory(os.path.join(app.root_path, "Pic"), filename)

if __name__ == "__main__":
    app.run(debug=True)
