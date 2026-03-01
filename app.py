from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer when the app starts
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ""
    
    # Check if the user submitted the form
    if request.method == 'POST':
        # Get the email text from the HTML form
        email_message = request.form['email_content']
        
        # Transform the text using the loaded vectorizer
        input_data_features = vectorizer.transform([email_message])
        
        # Make the prediction
        prediction = model.predict(input_data_features)
        
        # Format the result (1 = Ham, 0 = Spam based on your encoding)
        if prediction[0] == 1:
            prediction_text = "✅ This is a HAM (Safe) mail."
        else:
            prediction_text = "🚨 WARNING: This is a SPAM mail!"

    # Render the HTML page and pass the prediction result to it
    return render_template('index.html', prediction_result=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)