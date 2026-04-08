import os
from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
import bcrypt
from werkzeug.security import check_password_hash

db = SQLAlchemy()

def get_local_time():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='staff') # it_expert, doctor, staff
    blood_group = db.Column(db.String(5), nullable=False)
    email = db.Column(db.String(120), unique=True)
    phone_number = db.Column(db.String(20), unique=True)
    account_status = db.Column(db.String(20), default='Active')
    created_at = db.Column(db.DateTime, default=get_local_time)
    last_login = db.Column(db.DateTime)
    otp_secret = db.Column(db.String(6))
    otp_expiry = db.Column(db.DateTime)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    
    # Self-referential relationship
    doctor = db.relationship('User', remote_side=[id], backref='staff_members')

    def set_password(self, password):
        # Generate salt and hash the password
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        self.password_hash = hashed.decode('utf-8')

    def check_password(self, password):
        # Detect hash type: bcrypt hashes typically start with $2a$ or $2b$
        if self.password_hash.startswith('$2a$') or self.password_hash.startswith('$2b$'):
            try:
                return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))
            except ValueError:
                # In case of malformed bcrypt hash, fallback or fail
                return False
        # Fallback to Werkzeug check for legacy hashes (PBKDF2, etc.)
        return check_password_hash(self.password_hash, password)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    blood_group = db.Column(db.String(5), nullable=False)
    place = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), nullable=False, unique=True)
    created_at = db.Column(db.DateTime, default=get_local_time)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id', name='fk_patient_doctor_id'), nullable=True)
    staff_id = db.Column(db.Integer, db.ForeignKey('user.id', name='fk_patient_staff_id'), nullable=True)
    diagnoses = db.relationship('Diagnosis', backref='patient', lazy=True)
    doctor = db.relationship('User', foreign_keys=[doctor_id], backref=db.backref('assigned_patients', lazy=True))
    staff = db.relationship('User', foreign_keys=[staff_id], backref=db.backref('registered_patients', lazy=True))

class Diagnosis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    disease = db.Column(db.String(50), nullable=False)
    probability = db.Column(db.Float, nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    date = db.Column(db.DateTime, default=get_local_time)
    notes = db.Column(db.Text)

    doctor = db.relationship('User', backref=db.backref('diagnoses', lazy=True))

    @property
    def mask_path(self):
        """Derived path; Grad-CAM heatmaps are stored as gradcam_<stem>.jpg."""
        base = os.path.basename(self.image_path or '')
        stem, _ = os.path.splitext(base)
        return os.path.join('static', 'masks', f'gradcam_{stem}.jpg')

    @property
    def get_explanation(self):
        explanations = {
            "Normal": "No signs of diabetic retinopathy detected. Regular eye checkups are recommended.",
            "Mild": "Mild diabetic retinopathy indicates small changes in retinal blood vessels. Early monitoring is important.",
            "Moderate": "Moderate diabetic retinopathy shows increased damage to blood vessels in the retina. Medical consultation is recommended.",
            "Severe": "Severe diabetic retinopathy indicates significant blockage of retinal blood vessels. Immediate medical attention is advised.",
            "Proliferative DR": "Proliferative diabetic retinopathy is an advanced stage where abnormal blood vessels grow in the retina. Urgent treatment is required."
        }
        # Use default if disease name slightly mismatched
        return explanations.get(self.disease, "Regular medical consultation and eye checkups are recommended.")

    @property
    def get_lifestyle_advice(self):
        return [
            "Maintain healthy blood sugar levels",
            "Follow a balanced diet",
            "Exercise regularly",
            "Attend regular eye checkups"
        ]

class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id', name='fk_appointment_doctor_id'), nullable=True) # Usually required but keeping nullable for migration safety
    date = db.Column(db.Date, nullable=False)
    time = db.Column(db.Time, nullable=True) # Made nullable as it's no longer the primary scheduling field
    token_number = db.Column(db.Integer, nullable=False, default=1)
    status = db.Column(db.String(20), default='Waiting') # Changed default to 'Waiting' as requested
    created_at = db.Column(db.DateTime, default=get_local_time)
    
    patient_ref = db.relationship('Patient', backref=db.backref('appointments', lazy=True))
    doctor = db.relationship('User', backref=db.backref('appointments', lazy=True))

    __table_args__ = (
        db.UniqueConstraint('doctor_id', 'date', 'token_number', name='unique_appointment_token'),
    )
