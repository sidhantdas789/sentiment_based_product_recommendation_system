from flask import Flask, request, render_template
from model import SentimentRecommenderModel
import os


app = Flask(__name__)

sentiment_model = SentimentRecommenderModel()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def prediction():
    # get user from the html form
    user = request.form['userName']
    # convert text to lowercase
    user = user.lower()
    items = sentiment_model.get_sentiment_recommendations(user)

    if(items is not None):
        print(f"retrieving items....{len(items)}")
        print(items)
        # data=[items.to_html(classes="table-striped table-hover", header="true",index=False)
        return render_template("index.html", column_names=items.columns.values, row_data=list(items.values.tolist()), zip=zip)
    else:
        return render_template("index.html", message="User Name doesn't exists, No product recommendations at this point of time!")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
