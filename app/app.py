from flask import Flask, render_template, request, url_for, flash
import search
from markupsafe import escape
from forms import QueryForm

app = Flask(__name__)

app.config["SECRET_KEY"] = "710487b2a8794b0def310bb3e9ac71a437003816f46af86a5ea856cfbae91ed3"


@app.route("/", methods=["GET", "POST"])
def query():
    form = QueryForm()
    if form.validate_on_submit():
        time_req, result_json = search.get_results(form.query.data, return_n=-1)
    if request.method == "POST":
        return render_template("query.html", form=form, time_req=time_req, result_json=result_json, num_results=len(result_json))
    else:
        return render_template("query.html", form=form)



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)