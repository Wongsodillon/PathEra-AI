from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pandas as pd
from JobMatcher.job_matcher import JobRecommender

app = Flask(__name__)
CORS(app)  # Apply CORS to the Flask app

model = JobRecommender()

@app.route('/recommend', methods=['POST'])
def recommend():
    if request.is_json:
        try:
            user_data = request.get_json()
            user_data_df = pd.DataFrame([user_data])
            result, skill_matches = model.recommend_jobs(user_data_df)
            
            return jsonify({
                'success': True,
                'result': result.to_dict(orient='records'),
                'skill_matches': skill_matches.to_dict(orient='records')
            }), 200
        except FileNotFoundError as e:
            return jsonify({'success': False, 'error': f"File not found: {str(e)}"}), 500
        except pd.errors.EmptyDataError:
            return jsonify({'success': False, 'error': "The CSV file is empty."}), 500
        except Exception as e:
            print("Exception occurred:", e)
            return jsonify({'success': False, 'error': str(e)}), 500
    else:
        return jsonify({"error": "Request must be JSON"}), 400

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5020)
