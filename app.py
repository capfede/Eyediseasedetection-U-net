import os

from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from flask_mail import Mail, Message
import random
import string
from datetime import datetime, timedelta
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from functools import wraps
import numpy as np
from models import db, Patient, Diagnosis, Appointment, User, get_local_time
from unet_predict import generate_gradcam
from dr_predict import predict_dr_class
from pdf_reports import build_full_patient_report_pdf, build_single_diagnosis_pdf
import re
from datetime import datetime, date, time

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///eye_disease.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'supersecretkey'  # Needed for flash messages

# Flask-Mail Config
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', 'kesavcr0@gmail.com')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', 'qvdgoyeezskxgybl')
app.config['MAIL_DEFAULT_SENDER'] = app.config['MAIL_USERNAME']

mail = Mail(app)

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def create_admin():
    with app.app_context():
        if not User.query.filter_by(role='it_expert').first():
            admin = User(username='admin', role='it_expert', blood_group='O+')
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("Admin user created.")

            



def cleanup_old_appointments():
    """Automatically delete appointments older than 3 years."""
    try:
        cutoff_date = date.today() - timedelta(days=3*365)
        old_appointments = Appointment.query.filter(Appointment.date < cutoff_date).all()
        
        if old_appointments:
            count = len(old_appointments)
            for appt in old_appointments:
                db.session.delete(appt)
            db.session.commit()
            print(f"Cleanup: Deleted {count} appointments older than 3 years (before {cutoff_date}).")
    except Exception as e:
        db.session.rollback()
        print(f"Cleanup Error: {e}")

# Ensure database tables exist
with app.app_context():
    db.create_all()
    # Add new columns if they don't exist
    from sqlalchemy import text
    try:
        db.session.execute(text('ALTER TABLE user ADD COLUMN doctor_id INTEGER REFERENCES user(id)'))
        db.session.commit()
    except Exception:
        db.session.rollback()
        
    try:
        db.session.execute(text("ALTER TABLE user ADD COLUMN blood_group VARCHAR(5) NOT NULL DEFAULT 'O+'"))
        db.session.commit()
    except Exception:
        db.session.rollback()

    try:
        db.session.execute(text('ALTER TABLE patient ADD COLUMN staff_id INTEGER REFERENCES user(id)'))
        db.session.commit()
    except Exception:
        db.session.rollback()

    try:
        db.session.execute(text('ALTER TABLE diagnosis ADD COLUMN notes TEXT'))
        db.session.commit()
    except Exception:
        db.session.rollback()
        
    try:
        db.session.execute(text('ALTER TABLE appointment ADD COLUMN doctor_id INTEGER REFERENCES user(id)'))
        db.session.commit()
    except Exception:
        db.session.rollback()

    try:
        db.session.execute(text('ALTER TABLE appointment ADD COLUMN token_number INTEGER NOT NULL DEFAULT 1'))
        db.session.commit()
    except Exception:
        db.session.rollback()

    create_admin()
    cleanup_old_appointments()

# Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg', 'jpeg', 'png'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


# Email / 10-digit phone validation (users & patients)
RE_EMAIL = re.compile(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z0-9.-]+$')
RE_PHONE_10 = re.compile(r'^\d{10}$')


def valid_email_format(email):
    if email is None:
        return False
    s = str(email).strip()
    return bool(s and RE_EMAIL.fullmatch(s))


def valid_phone_10(phone):
    if phone is None:
        return False
    s = str(phone).strip()
    return bool(s and RE_PHONE_10.fullmatch(s))


BLOOD_GROUPS = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']


def valid_blood_group(blood_group):
    if blood_group is None:
        return False
    s = str(blood_group).strip()
    return s in BLOOD_GROUPS


def phone_optional_valid(phone):
    """Empty / whitespace-only is OK; otherwise must be exactly 10 digits."""
    if phone is None:
        return True
    s = str(phone).strip()
    if not s:
        return True
    return bool(RE_PHONE_10.fullmatch(s))


def valid_age(age):
    try:
        val = int(age)
        return 1 <= val <= 120
    except (ValueError, TypeError):
        return False


def role_required(role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated or current_user.role != role:
                flash('You do not have permission to access this page.', 'danger')
                return redirect(url_for('index'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route('/')
def index():
    # Landing page - redirect to login if not authenticated, otherwise to dashboard
    if current_user.is_authenticated:
        if current_user.role == 'it_expert':
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Redirect if already logged in
    if current_user.is_authenticated:
        if current_user.role == 'it_expert':
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            # Check if account is active
            if user.account_status != 'Active':
                flash('Your account has been deactivated. Please contact the administrator.', 'danger')
                return render_template('login.html')
            
            login_user(user)
            # Track last login
            user.last_login = get_local_time()
            db.session.commit()
            flash('Login Successful!', 'success')
            # Redirect to appropriate dashboard based on role
            if user.role == 'it_expert':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Redirect admin to admin dashboard 
    if current_user.role == 'it_expert':
        return redirect(url_for('admin_dashboard'))
    
    # Get stats for doctor/staff
    if current_user.role == 'staff':
        patient_count = Patient.query.filter_by(doctor_id=current_user.doctor_id).count() if current_user.doctor_id else 0
        appointment_count = Appointment.query.join(Patient).filter(Patient.doctor_id == current_user.doctor_id).count() if current_user.doctor_id else 0
    elif current_user.role == 'doctor':
        patient_count = Patient.query.filter_by(doctor_id=current_user.id).count()
        appointment_count = Appointment.query.join(Patient).filter(Patient.doctor_id == current_user.id).count()
    else:
        patient_count = Patient.query.count()
        appointment_count = Appointment.query.count()
        
    diagnosis_count = Diagnosis.query.count() if current_user.role == 'doctor' else 0
    staff_count = User.query.filter_by(role='staff').count() if current_user.role == 'doctor' else 0
    
    return render_template('unified_dashboard.html',
                           patient_count=patient_count,
                           appointment_count=appointment_count,
                           diagnosis_count=diagnosis_count,
                           staff_count=staff_count)

@app.route('/patient/search', methods=['POST'])
@login_required
def patient_search():
    # Patient search functionality
    if current_user.role not in ['doctor', 'staff', 'it_expert']:
        flash('Unauthorized access', 'danger')
        return redirect(url_for('dashboard'))
    
    patient_id = request.form.get('patient_id')
    if patient_id:
        patient = Patient.query.get(patient_id)
        if patient:
            if current_user.role == 'staff' and patient.doctor_id != current_user.doctor_id:
                flash('Unauthorized to view this patient.', 'danger')
                return redirect(url_for('dashboard'))
            if current_user.role == 'doctor' and patient.doctor_id != current_user.id:
                flash('Unauthorized to view this patient.', 'danger')
                return redirect(url_for('dashboard'))
            return redirect(url_for('patient_dashboard', patient_id=patient.id))
        else:
            flash('Patient ID not found!', 'danger')
    return redirect(url_for('dashboard'))

@app.route('/admin/dashboard')
@login_required
@role_required('it_expert')
def admin_dashboard():
    users = User.query.all()
    doctors = User.query.filter_by(role='doctor').order_by(User.username).all()
    doctor_count = User.query.filter_by(role='doctor').count()
    staff_count = User.query.filter_by(role='staff').count()
    patient_count = Patient.query.count()
    appointment_count = Appointment.query.count()
    
    return render_template('unified_dashboard.html', 
                           users=users,
                           doctors=doctors,
                           doctor_count=doctor_count,
                           staff_count=staff_count,
                           patient_count=patient_count,
                           appointment_count=appointment_count)

@app.route('/doctors')
@login_required
@role_required('it_expert')
def doctors_list():
    doctors = User.query.filter_by(role='doctor').order_by(User.id.desc()).all()
    return render_template('list_view.html', 
                           items=doctors,
                           list_type='doctors',
                           page_title='Doctors List')

@app.route('/staff')
@login_required
def staff_list():
    # Accessible to IT Expert and Doctor
    if current_user.role not in ['it_expert', 'doctor']:
        flash('Unauthorized access', 'danger')
        return redirect(url_for('dashboard'))
    
    if current_user.role == 'doctor':
        staff = User.query.filter_by(role='staff', doctor_id=current_user.id).order_by(User.id.desc()).all()
    else:
        staff = User.query.filter_by(role='staff').order_by(User.id.desc()).all()
    return render_template('list_view.html', 
                           items=staff,
                           list_type='staff',
                           page_title='Staff Members List')

@app.route('/patients')
@login_required
def patients_list():
    # Accessible to all authenticated users
    if current_user.role not in ['it_expert', 'doctor', 'staff']:
        flash('Unauthorized access', 'danger')
        return redirect(url_for('dashboard'))
    
    if current_user.role == 'doctor':
        patients = Patient.query.filter_by(doctor_id=current_user.id).order_by(Patient.id.desc()).all()
    elif current_user.role == 'staff':
        if not current_user.doctor_id:
            patients = []
        else:
            patients = Patient.query.filter_by(doctor_id=current_user.doctor_id).order_by(Patient.id.desc()).all()
    else:
        patients = Patient.query.order_by(Patient.id.desc()).all()
        
    return render_template('list_view.html', 
                           items=patients,
                           list_type='patients',
                           page_title='Patients List')


@app.route('/promote-staff', methods=['GET', 'POST'])
@login_required
@role_required('doctor')
def promote_staff():
    if request.method == 'POST':
        staff_id = request.form.get('staff_id')
        user = User.query.get(staff_id)
        if user and user.role == 'staff':
            user.role = 'doctor'
            db.session.commit()
            flash(f'User {user.username} has been promoted to Doctor.', 'success')
        else:
            flash('Invalid selection. Only staff members can be promoted.', 'danger')
        return redirect(url_for('promote_staff'))

    staff = User.query.filter_by(role='staff').order_by(User.id.desc()).all()
    return render_template('promote_staff.html', staff=staff)


@app.route('/admin/create_user', methods=['POST'])
@login_required
@role_required('it_expert')
def create_user():
    username = request.form['username']
    password = request.form['password']
    role = request.form['role']
    blood_group = (request.form.get('blood_group') or '').strip()
    email = (request.form.get('email') or '').strip()
    phone_number = (request.form.get('phone_number') or '').strip()
    doctor_id = request.form.get('doctor_id') if role == 'staff' else None

    if not blood_group:
        flash('Blood group is required', 'danger')
        return redirect(request.referrer or url_for('admin_dashboard'))
    if not valid_blood_group(blood_group):
        flash('Invalid blood group selected. Choose one of A+, A-, B+, B-, AB+, AB-, O+, O-.', 'danger')
        return redirect(request.referrer or url_for('admin_dashboard'))
    
    if not valid_email_format(email):
        flash('Invalid email format.', 'danger')
        return redirect(request.referrer or url_for('admin_dashboard'))
    if not phone_optional_valid(phone_number):
        flash('Invalid phone number. Enter 10 digits only.', 'danger')
        return redirect(request.referrer or url_for('admin_dashboard'))
    
    if User.query.filter_by(username=username).first():
        flash('Username already exists', 'danger')
    elif email and User.query.filter_by(email=email).first(): # Check email uniqueness
        flash('Email already registered', 'danger')
    elif phone_number and User.query.filter_by(phone_number=phone_number).first(): # Check phone uniqueness
        flash('Phone number already registered', 'danger')
    else:
        new_user = User(username=username, role=role, blood_group=blood_group, email=email, phone_number=phone_number or None)
        if role == 'staff' and doctor_id:
            new_user.doctor_id = doctor_id
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash(f'User {username} created successfully', 'success')
    
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/view_user/<int:user_id>')
@login_required
@role_required('it_expert')
def view_user(user_id):
    user = User.query.get_or_404(user_id)
    return render_template('view_user_profile.html', user=user)

@app.route('/admin/edit_user/<int:user_id>', methods=['GET', 'POST'])
@login_required
@role_required('it_expert')
def admin_edit_user(user_id):
    user = User.query.get_or_404(user_id)
    
    if request.method == 'POST':
        new_username = request.form.get('username')
        new_email = (request.form.get('email') or '').strip()
        new_phone = (request.form.get('phone_number') or '').strip()
        new_role = request.form.get('role')
        new_blood_group = (request.form.get('blood_group') or '').strip()
        if not new_role:
            # Role <select> is disabled when editing own account; preserve existing role
            new_role = user.role

        if not new_blood_group:
            flash('Blood group is required', 'danger')
            return redirect(url_for('admin_edit_user', user_id=user_id))
        if not valid_blood_group(new_blood_group):
            flash('Invalid blood group selected. Choose one of A+, A-, B+, B-, AB+, AB-, O+, O-.', 'danger')
            return redirect(url_for('admin_edit_user', user_id=user_id))
        
        if new_email and not valid_email_format(new_email):
            flash('Invalid email format.', 'danger')
            return redirect(url_for('admin_edit_user', user_id=user_id))
        if not phone_optional_valid(request.form.get('phone_number')):
            flash('Invalid phone number. Enter 10 digits only.', 'danger')
            return redirect(url_for('admin_edit_user', user_id=user_id))
        
        # Check username uniqueness
        if new_username != user.username:
            if User.query.filter_by(username=new_username).first():
                flash('Username already exists', 'danger')
                return redirect(url_for('admin_edit_user', user_id=user_id))
        
        # Check email uniqueness (only when setting a non-empty email)
        if new_email and new_email != user.email:
            if User.query.filter_by(email=new_email).first():
                flash('Email already registered', 'danger')
                return redirect(url_for('admin_edit_user', user_id=user_id))
        
        # Check phone uniqueness
        if new_phone and new_phone != user.phone_number:
            if User.query.filter_by(phone_number=new_phone).first():
                flash('Phone number already registered', 'danger')
                return redirect(url_for('admin_edit_user', user_id=user_id))
        
        # Update user (staff–doctor assignment: only for staff; empty = no doctor)
        user.username = new_username
        user.email = new_email or None
        user.phone_number = new_phone or None
        user.blood_group = new_blood_group
        user.role = new_role
        if new_role == 'staff':
            raw_doc = request.form.get('doctor_id')
            if raw_doc is None or str(raw_doc).strip() == '':
                user.doctor_id = None
            else:
                try:
                    did = int(raw_doc)
                except (TypeError, ValueError):
                    flash('Invalid doctor selection.', 'danger')
                    return redirect(url_for('admin_edit_user', user_id=user_id))
                doc_user = User.query.get(did)
                if not doc_user or doc_user.role != 'doctor':
                    flash('Selected user is not a valid doctor.', 'danger')
                    return redirect(url_for('admin_edit_user', user_id=user_id))
                user.doctor_id = did
        else:
            user.doctor_id = None
        db.session.commit()
        
        flash(f'User {user.username} updated successfully', 'success')
        return redirect(url_for('view_user', user_id=user_id))
    
    doctors = User.query.filter_by(role='doctor').order_by(User.username).all()
    return render_template('edit_user.html', user=user, doctors=doctors)

@app.route('/admin/toggle_status/<int:user_id>', methods=['POST'])
@login_required
@role_required('it_expert')
def admin_toggle_status(user_id):
    user = User.query.get_or_404(user_id)
    
    # Prevent admin from deactivating themselves
    if user.id == current_user.id:
        flash('You cannot deactivate your own account', 'danger')
        return redirect(url_for('admin_dashboard'))
    
    # Toggle status
    if user.account_status == 'Active':
        user.account_status = 'Deactivated'
        flash(f'User {user.username} has been deactivated', 'warning')
    else:
        user.account_status = 'Active'
        flash(f'User {user.username} has been activated', 'success')
    
    db.session.commit()
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
@role_required('it_expert')
def admin_delete_user(user_id):
    user = User.query.get_or_404(user_id)
    
    # Prevent admin from deleting themselves
    if user.id == current_user.id:
        flash('You cannot delete your own account', 'danger')
        return redirect(url_for('admin_dashboard'))
    
    # Check if user has diagnoses (safety check)
    if user.diagnoses:
        flash(f'Cannot delete user {user.username}: User has {len(user.diagnoses)} diagnosis records. Deactivate instead.', 'danger')
        return redirect(url_for('admin_dashboard'))
    
    # Set assigned patients doctor_id to None
    if user.role == 'doctor':
        Patient.query.filter_by(doctor_id=user.id).update({Patient.doctor_id: None})
    elif user.role == 'staff':
        Patient.query.filter_by(staff_id=user.id).update({Patient.staff_id: None})
    
    username = user.username
    db.session.delete(user)
    db.session.commit()
    
    flash(f'User {username} has been deleted', 'info')
    return redirect(url_for('admin_dashboard'))


@app.route('/register', methods=['GET', 'POST'])
@login_required
def register():
    if current_user.role not in ['staff', 'it_expert']: 
         flash('Permission denied', 'danger')
         return redirect(url_for('index'))
             
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        gender = request.form['gender']
        blood_group = (request.form.get('blood_group') or '').strip()
        place = request.form['place']
        phone = (request.form.get('phone') or '').strip()

        if not blood_group:
            flash('Blood group is required', 'danger')
            return redirect(url_for('register'))
        if not valid_blood_group(blood_group):
            flash('Invalid blood group selected. Choose one of A+, A-, B+, B-, AB+, AB-, O+, O-.', 'danger')
            return redirect(url_for('register'))

        if not valid_age(age):
            flash('Enter a valid age', 'danger')
            return redirect(url_for('register'))

        if not valid_phone_10(phone):
            flash('Invalid phone number. Enter 10 digits only.', 'danger')
            return redirect(url_for('register'))
        
        if Patient.query.filter_by(phone=phone).first():
            flash('Phone number already registered', 'danger')
            return redirect(url_for('register'))
        
        if current_user.role == 'staff':
            doctor_id = current_user.doctor_id
            staff_id = current_user.id
            if not doctor_id:
                flash('You are not assigned to any doctor. Cannot register patients.', 'danger')
                return redirect(url_for('dashboard'))
        else:
            doctor_id = request.form.get('doctor_id')
            staff_id = request.form.get('staff_id') or None
            
            if doctor_id:
                doctor_check = User.query.get(doctor_id)
                if not doctor_check or doctor_check.role != 'doctor':
                    flash('Invalid doctor selected.', 'danger')
                    return redirect(url_for('register'))
            else:
                doctor_id = None
                
            if staff_id:
                staff_check = User.query.get(staff_id)
                if not staff_check or staff_check.role != 'staff' or str(staff_check.doctor_id) != str(doctor_id):
                    flash('Invalid staff selected or staff does not belong to the selected doctor.', 'danger')
                    return redirect(url_for('register'))
            else:
                staff_id = None
        
        new_patient = Patient(name=name, age=age, gender=gender, blood_group=blood_group, place=place, phone=phone, doctor_id=doctor_id, staff_id=staff_id)
        db.session.add(new_patient)
        db.session.commit()
        
        flash(f'Patient Registered Successfully! ID: {new_patient.id}', 'success')
        return redirect(url_for('patient_dashboard', patient_id=new_patient.id))
        
    doctors = User.query.filter_by(role='doctor').order_by(User.username).all()
    all_staff = User.query.filter_by(role='staff').order_by(User.username).all()
    return render_template('register.html', doctors=doctors, all_staff=all_staff)
    
@app.route('/patient/<int:patient_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_patient(patient_id):
    if current_user.role not in ['staff', 'it_expert']:
         flash('Permission denied. Only staff can edit patients.', 'danger')
         return redirect(url_for('patient_dashboard', patient_id=patient_id))
         
    patient = Patient.query.get_or_404(patient_id)
    if current_user.role == 'staff' and patient.doctor_id != current_user.doctor_id:
         flash('Unauthorized. You can only edit patients assigned to your doctor.', 'danger')
         return redirect(url_for('patient_dashboard', patient_id=patient_id))
         
    if request.method == 'POST':
        phone = (request.form.get('phone') or '').strip()
        age = request.form.get('age')

        if not valid_age(age):
            flash('Enter a valid age', 'danger')
            return redirect(url_for('edit_patient', patient_id=patient_id))

        if not valid_phone_10(phone):
            flash('Invalid phone number. Enter 10 digits only.', 'danger')
            return redirect(url_for('edit_patient', patient_id=patient_id))
        
        # Check phone uniqueness for patient
        if phone != patient.phone:
            if Patient.query.filter_by(phone=phone).first():
                flash('Phone number already registered', 'danger')
                return redirect(url_for('edit_patient', patient_id=patient_id))

        blood_group = (request.form.get('blood_group') or '').strip()
        if not blood_group:
            flash('Blood group is required', 'danger')
            return redirect(url_for('edit_patient', patient_id=patient_id))
        if not valid_blood_group(blood_group):
            flash('Invalid blood group selected. Choose one of A+, A-, B+, B-, AB+, AB-, O+, O-.', 'danger')
            return redirect(url_for('edit_patient', patient_id=patient_id))

        patient.name = request.form['name']
        patient.age = int(age)
        patient.gender = request.form['gender']
        patient.blood_group = blood_group
        patient.place = request.form['place']
        patient.phone = phone
        
        if current_user.role == 'staff':
            pass # Keep existing doctor_id
        else:
            doctor_id = request.form.get('doctor_id')
            staff_id = request.form.get('staff_id') or None
            
            if doctor_id:
                doctor_check = User.query.get(doctor_id)
                if not doctor_check or doctor_check.role != 'doctor':
                    flash('Invalid doctor selected.', 'danger')
                    return redirect(url_for('edit_patient', patient_id=patient_id))
                patient.doctor_id = doctor_id
            else:
                patient.doctor_id = None
                
            if staff_id:
                staff_check = User.query.get(staff_id)
                if not staff_check or staff_check.role != 'staff' or str(staff_check.doctor_id) != str(doctor_id):
                    flash('Invalid staff selected or staff does not belong to the selected doctor.', 'danger')
                    return redirect(url_for('edit_patient', patient_id=patient_id))
                patient.staff_id = staff_id
            else:
                patient.staff_id = None
            
        db.session.commit()
        flash('Patient details updated successfully!', 'success')
        return redirect(url_for('patient_dashboard', patient_id=patient.id))
        
    doctors = User.query.filter_by(role='doctor').order_by(User.username).all()
    all_staff = User.query.filter_by(role='staff').order_by(User.username).all()
    return render_template('edit_patient.html', patient=patient, doctors=doctors, all_staff=all_staff)

@app.route('/diagnosis/patients')
@login_required
@role_required('doctor')
def diagnosis_patients():
    # Show list of all patients assigned to the logged-in doctor
    patients = Patient.query.filter_by(doctor_id=current_user.id).order_by(Patient.id.desc()).all()
    return render_template('diagnosis_patients.html', patients=patients)

@app.route('/patient/<int:patient_id>', methods=['GET'])
@login_required
def patient_dashboard(patient_id):
    # Staff and Doctor can view dashboard, but content is filtered in template
    if current_user.role not in ['staff', 'doctor', 'it_expert']:
        flash('Unauthorized', 'danger')
        return redirect(url_for('index'))
    patient = Patient.query.get_or_404(patient_id)
    if current_user.role == 'staff' and patient.doctor_id != current_user.doctor_id:
        flash('Unauthorized to view this patient.', 'danger')
        return redirect(url_for('dashboard'))
    if current_user.role == 'doctor' and patient.doctor_id != current_user.id:
        flash('Unauthorized to view this patient.', 'danger')
        return redirect(url_for('dashboard'))
    return render_template('patient_dashboard.html', patient=patient)

@app.route('/predict/<int:patient_id>', methods=['POST'])
@login_required
@role_required('doctor')
def predict(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    
    file = request.files.get('filename')
    if file and allowed_file(file.filename):
        filename = f"{patient_id}_{int(get_local_time().timestamp())}_{file.filename}"
        file_path = os.path.join('static/images', filename)
        file.save(file_path)

        # 1. Grad-CAM attention map (replaces U-Net segmentation)
        mask_path = generate_gradcam(file_path)

        # 2. DR Classification (for actual diagnosis)
        label, confidence = predict_dr_class(file_path)

        notes_raw = (request.form.get('notes') or '').strip()
        notes_val = notes_raw if notes_raw else None

        # Save Diagnosis to DB
        new_diagnosis = Diagnosis(
            patient_id=patient.id,
            doctor_id=current_user.id,
            disease=label,
            probability=confidence,
            image_path=file_path,
            notes=notes_val,
        )
        db.session.add(new_diagnosis)
        db.session.commit()
        
        flash('Diagnosis added successfully!', 'success')
        return render_template('patient_dashboard.html', 
                               patient=patient, 
                               diagnosis_result=new_diagnosis, 
                               mask_path=mask_path,
                               notes=notes_val)
    else:
        flash('Invalid file or upload failed.', 'danger')
        return redirect(url_for('patient_dashboard', patient_id=patient_id))


def _can_view_diagnosis_report(diagnosis):
    """Same access rules as patient dashboard for the diagnosis's patient."""
    patient = diagnosis.patient
    if current_user.role not in ['staff', 'doctor', 'it_expert']:
        return False
    if current_user.role == 'staff' and patient.doctor_id != current_user.doctor_id:
        return False
    if current_user.role == 'doctor' and patient.doctor_id != current_user.id:
        return False
    return True


@app.route('/diagnosis/<int:id>')
@login_required
def diagnosis_report(id):
    diagnosis = Diagnosis.query.get_or_404(id)
    if not _can_view_diagnosis_report(diagnosis):
        flash('Unauthorized to view this report.', 'danger')
        return redirect(url_for('dashboard'))
    return render_template(
        'diagnosis_report.html',
        diagnosis=diagnosis,
        patient=diagnosis.patient,
    )


def _staff_can_download_patient_pdf(patient):
    if current_user.role != 'staff':
        return False
    if not current_user.doctor_id or patient.doctor_id != current_user.doctor_id:
        return False
    return True


def _send_pdf_attachment(buffer, filename):
    """send_file with download_name (Flask 2+) or attachment_filename (older)."""
    buffer.seek(0)
    try:
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename,
        )
    except TypeError:
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            attachment_filename=filename,
        )


@app.route('/download_report/<int:id>')
@login_required
def download_report(id):
    if current_user.role != 'staff':
        return 'Unauthorized Access', 403
    diagnosis = Diagnosis.query.get_or_404(id)
    patient = diagnosis.patient
    if not _staff_can_download_patient_pdf(patient):
        return 'Unauthorized Access', 403
    buffer = build_single_diagnosis_pdf(patient, diagnosis, app.root_path)
    return _send_pdf_attachment(buffer, f'diagnosis_report_{id}.pdf')


@app.route('/download_patient_report/<int:patient_id>')
@login_required
def download_patient_report(patient_id):
    if current_user.role != 'staff':
        return 'Unauthorized Access', 403
    patient = Patient.query.get_or_404(patient_id)
    if not _staff_can_download_patient_pdf(patient):
        return 'Unauthorized Access', 403
    diagnoses = Diagnosis.query.filter_by(patient_id=patient.id).all()
    diagnoses.sort(key=lambda d: (d.date or datetime.min, d.id), reverse=True)
    buffer = build_full_patient_report_pdf(patient, diagnoses, app.root_path)
    return _send_pdf_attachment(buffer, f'patient_{patient_id}_full_report.pdf')


@app.route('/appointments', methods=['GET'])
@login_required
def appointments():
    if current_user.role not in ['staff', 'doctor', 'it_expert']:
        flash('Unauthorized', 'danger')
        return redirect(url_for('index'))
    view_type = request.args.get('view_type', 'all')
    filter_date = request.args.get('date')
    
    if current_user.role == 'staff':
        if not current_user.doctor_id:
            query = Appointment.query.filter(False) # No doctor assigned means no patients/appointments
        else:
            query = Appointment.query.join(Patient).filter(Patient.doctor_id == current_user.doctor_id)
    elif current_user.role == 'doctor':
        query = Appointment.query.join(Patient).filter(Patient.doctor_id == current_user.id)
    else:
        query = Appointment.query
    
    if view_type == 'today':
        query = query.filter(Appointment.date == date.today())
    elif view_type == 'upcoming':
        query = query.filter(Appointment.date > date.today())
    elif filter_date:
        try:
            search_date = datetime.strptime(filter_date, '%Y-%m-%d').date()
            query = query.filter(Appointment.date == search_date)
        except ValueError:
            flash('Invalid date format', 'danger')
            
    appointments = query.order_by(Appointment.date, Appointment.token_number).all()
    
    return render_template('appointments.html', 
                           appointments=appointments, 
                           filter_date=filter_date,
                           view_type=view_type,
                           today=date.today().strftime('%Y-%m-%d'))

@app.route('/appointments/today', methods=['GET'])
@login_required
def today_appointments():
    today = date.today()
    return redirect(url_for('appointments', date=today.isoformat()))

@app.route('/book_appointment', methods=['GET', 'POST'])
@login_required
def book_appointment():
    if current_user.role not in ['staff', 'it_expert']: 
         flash('Permission denied', 'danger')
         return redirect(url_for('dashboard'))
    if request.method == 'POST':
        patient_id = request.form.get('patient_id')
        appt_date_str = request.form.get('appointment_date')
        appt_time_str = request.form.get('appointment_time')
        
        if not patient_id or not appt_date_str:
            flash('Patient and Date are required', 'danger')
            return redirect(url_for('book_appointment'))
            
        try:
            appt_date = datetime.strptime(appt_date_str, '%Y-%m-%d').date()
            if appt_date < date.today():
                flash('Cannot book appointment in the past.', 'danger')
                return redirect(url_for('book_appointment'))
            
            patient = Patient.query.get(patient_id)
            if not patient:
                flash('Invalid patient selected.', 'danger')
                return redirect(url_for('book_appointment'))
            
            doctor_id = patient.doctor_id
            if not doctor_id:
                flash('Patient has no assigned doctor. Cannot book appointment.', 'danger')
                return redirect(url_for('book_appointment'))

            # Token generation logic
            last_appt = Appointment.query.filter_by(doctor_id=doctor_id, date=appt_date).order_by(Appointment.token_number.desc()).first()
            token_number = (last_appt.token_number + 1) if last_appt else 1
                
            new_appt = Appointment(
                patient_id=patient_id,
                doctor_id=doctor_id,
                date=appt_date,
                token_number=token_number,
                status='Waiting'
            )
            db.session.add(new_appt)
            db.session.commit()
            flash(f'Appointment booked successfully! Your Token Number: {token_number} (Doctor: Dr. {new_appt.doctor.username}, Date: {appt_date_str})', 'success')
            return redirect(url_for('appointments', date=appt_date_str))
            
        except ValueError:
            flash('Invalid date/time format', 'danger')
            
    if current_user.role == 'staff':
        patients = Patient.query.filter_by(doctor_id=current_user.doctor_id).order_by(Patient.name).all()
    else:
        patients = Patient.query.order_by(Patient.name).all()
    # Pre-select patient if patient_id is passed in args
    selected_patient_id = request.args.get('patient_id')
    return render_template('book_appointment.html', patients=patients, selected_patient_id=selected_patient_id)

@app.route('/appointment/<int:id>/complete', methods=['POST'])
@login_required
def complete_appointment(id):
    if current_user.role != 'staff':
        flash('Only staff can mark appointments as completed.', 'danger')
        return redirect(url_for('appointments'))
        
    appt = Appointment.query.get_or_404(id)
    # Security check: staff can only complete appointments for their assigned doctor
    if current_user.doctor_id and appt.doctor_id != current_user.doctor_id:
        flash('Unauthorized to manage this appointment.', 'danger')
        return redirect(url_for('appointments'))
        
    appt.status = 'Completed'
    db.session.commit()
    flash(f'Appointment for {appt.patient_ref.name} (Token #{appt.token_number}) marked as Completed.', 'success')
    return redirect(url_for('appointments', date=appt.date.strftime('%Y-%m-%d')))

# OTP Helper
def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

def send_otp_email(user, purpose):
    otp = generate_otp()
    user.otp_secret = otp
    user.otp_expiry = get_local_time() + timedelta(minutes=5)
    db.session.commit()
    
    msg = Message(f'Your OTP for {purpose}', recipients=[user.email])
    msg.body = f'Your OTP is: {otp}. It expires in 5 minutes.'
    try:
        mail.send(msg)
        print(f"DEBUG: Sent OTP {otp} to {user.email}") # For local testing if mail fails
    except Exception as e:
        print(f"Error sending email: {e}")
        flash('Error sending email. Check console for debug OTP.', 'warning')
        print(f"DEBUG: OTP (Fallback) for {user.email}: {otp}")

# Forgot Password Routes
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = (request.form.get('email') or '').strip()
        if not valid_email_format(email):
            flash('Invalid email format.', 'danger')
            return redirect(url_for('forgot_password'))
        user = User.query.filter_by(email=email).first()
        if user:
            send_otp_email(user, 'Password Reset')
            session['reset_email'] = email
            session['otp_purpose'] = 'reset_password'
            flash('OTP sent to your email', 'info')
            return redirect(url_for('verify_otp'))
        else:
            flash('Email not found', 'danger')
    return render_template('forgot_password.html')

@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    if request.method == 'POST':
        otp_input = request.form['otp']
        purpose = session.get('otp_purpose')
        
        if purpose == 'reset_password':
            email = session.get('reset_email')
            user = User.query.filter_by(email=email).first()
        elif purpose in ['change_username', 'change_password']:
            user = current_user
        else:
            flash('Invalid session', 'danger')
            return redirect(url_for('login'))

        if user and user.otp_secret == otp_input and user.otp_expiry > get_local_time():
            user.otp_secret = None # Clear OTP
            user.otp_expiry = None
            db.session.commit()
            session['otp_verified'] = True
            
            if purpose == 'reset_password':
                return redirect(url_for('reset_password'))
            elif purpose == 'change_username':
                 # Apply change immediately or redirect? Let's apply valid pending change
                 new_username = session.get('pending_username')
                 if new_username:
                     user.username = new_username
                     db.session.commit()
                     flash('Username updated successfully!', 'success')
                     return redirect(url_for('profile'))
            elif purpose == 'change_password':
                 new_password = session.get('pending_password')
                 if new_password:
                     user.set_password(new_password)
                     db.session.commit()
                     flash('Password updated successfully!', 'success')
                     return redirect(url_for('profile'))
        else:
            flash('Invalid or Expired OTP', 'danger')
    return render_template('verify_otp.html')

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if not session.get('otp_verified') or session.get('otp_purpose') != 'reset_password':
        return redirect(url_for('forgot_password'))
        
    if request.method == 'POST':
        password = request.form['password']
        confirm = request.form['confirm_password']
        if password != confirm:
            flash('Passwords do not match', 'danger')
        else:
            email = session.get('reset_email')
            user = User.query.filter_by(email=email).first()
            if user:
                user.set_password(password)
                db.session.commit()
                flash('Password reset successful. Please Login.', 'success')
                session.clear()
                return redirect(url_for('login'))
    return render_template('reset_password.html')

# Profile Routes
@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')

@app.route('/profile/change_username', methods=['POST'])
@login_required
def change_username():
    new_username = request.form['new_username']
    if User.query.filter_by(username=new_username).first():
        flash('Username already exists', 'danger')
        return redirect(url_for('profile'))
    
    session['pending_username'] = new_username
    session['otp_purpose'] = 'change_username'
    if not current_user.email:
         flash('No email registered for this account. Cannot verify.', 'danger')
         return redirect(url_for('profile'))
         
    send_otp_email(current_user, 'Username Change')
    flash('OTP sent to your email to confirm username change', 'info')
    return redirect(url_for('verify_otp'))

@app.route('/profile/change_password', methods=['POST'])
@login_required
def change_password():
    new_password = request.form['new_password']
    session['pending_password'] = new_password
    session['otp_purpose'] = 'change_password'
    
    if not current_user.email:
         flash('No email registered for this account. Cannot verify.', 'danger')
         return redirect(url_for('profile'))

    send_otp_email(current_user, 'Password Change')
    flash('OTP sent to your email to confirm password change', 'info')
    return redirect(url_for('verify_otp'))

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True, port=8000)