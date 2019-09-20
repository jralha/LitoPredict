from flask import Flask, request, render_template, Response, make_response, redirect, session, abort, flash
import os
import pandas as pd
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer

app = Flask(__name__)
app.secret_key = os.urandom(12)

@app.route('/')
def home():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return redirect("/predict")

@app.route('/login', methods=['POST'])
def do_admin_login():
    if request.form['password'] == 'petrec@123' and request.form['username'] == 'petrec':
        session['logged_in'] = True
    else:
        flash('Wrong Password!')
    return redirect("/predict")

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if session.get('logged_in'):
        print(request.method)
        if request.method == 'POST':
            df = pd.read_csv(request.files.get('file'))
            col_dtc = int(request.form["DTc"])-1
            col_dts = int(request.form["DTs"])-1
            col_rhob = int(request.form["RHOB"])-1
            col_gr = int(request.form["GR"])-1
            head_lines = int(request.form["header"])-1
            col_md = request.form["mds"]

            data0 = df.iloc[head_lines:,:]
            data = data0.iloc[:,[col_dtc,col_dts,col_rhob,col_gr]]

            imp = SimpleImputer()
            polyfeat = PolynomialFeatures()
            data_imp = imp.fit_transform(data)
            data_feat = polyfeat.fit_transform(data_imp)
            data_df = pd.DataFrame(data_feat)

            data_df.columns = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14']

            model = pickle.load(open("model.pkl",'rb'))

            preds = model.predict(data_df)
            probs = model.predict_proba(data_df)
            prob0 = probs.T[0]
            prob1 = probs.T[1]
            prob_f = [0] * len(preds)
            for p in range(len(prob_f)):
                if prob0[p] > 0.5:
                    prob_f[p] = prob0[p]
                else:
                    prob_f[p] = prob1[p]

            data['Pred'] = pd.Series(preds)
            data['Pred'] = data['Pred'].map({0:'Itapema',1:'Barra Velha'})
            data['Prob'] = pd.Series(prob_f)

            out = data
            
            if not col_md == "":
                
                col_md = int(col_md)-1
                data['MD'] = data0.iloc[:,col_md]
                out.set_index('MD')
            

            resp = make_response(out.to_csv())
            resp.headers["Content-Disposition"] = "attachment; filename=pred.csv"
            resp.headers["Content-Type"] = "text/csv"
            return resp
        else:
            return render_template('upload.html')
    else:
        return redirect("/login")

if __name__ == "__main__":
    app.run(debug=True)





