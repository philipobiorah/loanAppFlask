from flask import Flask, render_template, redirect, request, url_for
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder

#load our model
model = joblib.load(r'RanFor_model.pkl')
# stdscaler = joblib.load(r'my_scaler.save')

#configure app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        #Getting the data from the form
        firstname = request.form.get("firstname")
        lastname = request.form.get("lastname")
        income_level = request.form.get("incomelevel")
        loanamount = request.form.get("loanamount")
        credit_history = request.form.get("points")
        gender = request.form.get("gender")
        married = request.form.get("married")
        university_staff = request.form.get("unistaffs")
        union_member = request.form.get("member")
        guarantor = request.form.get("guarans")
        guarantor_contribution = request.form.get("guarancons")
        pledge = request.form.get("pledge") 
        #creating a josn object to hold the data from the form
        input_data=[{
            'income_level':income_level,
            'loanamount':loanamount,
            'credit_history':credit_history,
            'gender':gender,
            'married':married,
            'university_staff':university_staff,
            'union_member':union_member,
            'guarantor':guarantor,
            'guarantor_contribution':guarantor_contribution,
            'pledge':pledge }]
        data=pd.DataFrame(input_data)
        le = LabelEncoder() 
        categorical_columns=['gender', 'married', 'guarantor', 'guarantor_contribution', 'university_staff']
        data['income_level'] = data['income_level'].astype(np.int64)
        data['loanamount'] = data['loanamount'].astype(np.int64)
        data['credit_history'] = data['credit_history'].astype(np.int64)
        data['union_member'] = data['union_member'].astype(np.int64)
        data['pledge'] = data['pledge'].astype(np.int64)
        data[categorical_columns]=data[categorical_columns].apply(le.fit_transform)
        data[categorical_columns]=data[categorical_columns].astype('object')

        # test_new = stdscaler.fit_transform(data) 
        pred = model.predict(data)
        
        if pred==1:
            result = "Approved!"
        elif(pred==0):
            result = "Declined!"
        # return render_template("index.html", firstname=firstname, lastname=lastname )   
        return render_template('results.html', res=result, firstname=firstname, lastname=lastname) 
    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST) 


# def enval(df):
#     var_mod = ['Gender', 'Married', 'Guarantor', 'Guarantor_Contribution', 'University_Staff']
#     le = LabelEncoder()
#     for i in var_mod:
#         df[i] = le.fit_transform(df[i])
#         return df
  
# def approvereject(unit):
#     try:
#         model = joblib.load(r'KNN_model.pkl')
#         stdscaler = joblib.load(r'my_scaler.save')
#         test = stdscaler.fit_transform(unit)
#         pred = model.predict(test)
#         y_pred = list(map(lambda x: 'Approved' if x == 1 else 'Declined', pred))
#         return render_template("success.html", firstname=firstname, lastname=lastname) 
#         return 'Your Status is {}'.format(val)
#     except ValueError as e:
#         return Response(e.args[0], status.HTTP_400_BAD_REQUEST)    
    
if __name__=='__main__':
    app.run(port=3000, debug=True)   