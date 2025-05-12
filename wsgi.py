import logging

# Import the Flask app object created in app.py
try:
    # The critical part: app.py should define the 'app' variable
    from app import app
    logging.info("WSGI entry point: Successfully imported 'app' from app.py")
except ImportError as e:
     logging.critical(f"WSGI entry point: Failed to import 'app' from app.py: {e}", exc_info=True)
     # If app cannot be imported, WSGI server will likely fail.
     # You might want to create a dummy app here to return an error message
     from flask import Flask, jsonify
     app = Flask(__name__)
     @app.route('/', defaults={'path': ''})
     @app.route('/<path:path>')
     def import_error_handler(path):
         return jsonify({"status": "error", "message": "Backend application failed to initialize due to import error."}), 500
except Exception as e:
     # Catch other potential errors during import/initialization in app.py
     logging.critical(f"WSGI entry point: An unexpected error occurred during app import/initialization: {e}", exc_info=True)
     from flask import Flask, jsonify
     app = Flask(__name__)
     @app.route('/', defaults={'path': ''})
     @app.route('/<path:path>')
     def general_error_handler(path):
         return jsonify({"status": "error", "message": f"Backend application failed to initialize: {e}"}), 500


# Gunicorn (or other WSGI servers) will look for the 'app' variable by default.
# The logging configuration should ideally happen within app.py so it's
# configured when the app object is created.

# Example Gunicorn command:
# gunicorn --workers 2 --bind 0.0.0.0:8000 wsgi:app

# --- End of wsgi.py ---
