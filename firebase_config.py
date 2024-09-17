# firebase_config.py

import firebase_admin
from firebase_admin import credentials, firestore

# Initialize the app with a service account, granting admin privileges
cred = credentials.Certificate('path/to/serviceAccountKey.json')
firebase_app = firebase_admin.initialize_app(cred)

# Initialize Firestore DB
db = firestore.client()
