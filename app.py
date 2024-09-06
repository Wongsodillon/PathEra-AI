import os
from flask import Flask, request, jsonify
from flask_cors import CORS 
import pandas as pd
from JobMatcher.job_matcher import JobRecommender
from InterviewSimulation.evaluator import Evaluator

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://pathera.vercel.app/"}})

db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),  
    'user': os.getenv('DB_USER', 'root'),     
    'password': os.getenv('DB_PASSWORD', ''),   
    'database': os.getenv('DB_NAME', 'pathera'), 
    'port': int(os.getenv('DB_PORT', 3306))     
}

model = JobRecommender()
# evaluator = Evaluator(db_config=db_config, model_name="./InterviewSimulation/model/multiple_negatives_11")
evaluator = Evaluator(db_config=db_config)

@app.route('/recommend', methods=['POST'])
def recommend():
    if request.is_json:
        try:
            user_data = request.get_json()
            user_data_df = pd.DataFrame([user_data])
            print(user_data_df['skills'])
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

@app.route('/evaluate', methods=["POST"]) 
def evaluate():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json()
    question_id = data.get('question_id')
    user_answer = data.get('user_answer')
    if not question_id or not user_answer:
        return jsonify({"error": "Missing required fields"}), 400
    if evaluator.check_question_exists(question_id) is False:
        return jsonify({"error": "Question does not exist"}), 400
    try:
        score = evaluator.evaluate(question_id, user_answer)
        return jsonify({"success": True, "score": score }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"success": True}), 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=int(os.getenv('PORT', 5020)))
