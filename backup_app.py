# @app.route('/predict', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         df = pd.read_csv(request.files.get('file'))
#         col_dtc = int(request.form["DTc"])-1
#         col_dts = int(request.form["DTs"])-1
#         col_rhob = int(request.form["RHOB"])-1
#         col_gr = int(request.form["GR"])-1
#         head_lines = int(request.form["header"])

#         data0 = df.iloc[head_lines:,:]
#         data = data0.iloc[:,[col_dtc,col_dts,col_rhob,col_gr]]

#         #Data transformation like in model
#         imp = SimpleImputer()
#         polyfeat = PolynomialFeatures()
#         data_imp = imp.fit_transform(data)
#         data_feat = polyfeat.fit_transform(data_imp)
#         data_df = pd.DataFrame(data_feat)

#         #Columns used in XGBoost model
#         data_df.columns = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14']

#         model = pickle.load(open("model.pkl",'rb'))

#         preds = model.predict(data_df)

#         data['Pred'] = pd.Series(preds)

#         resp = make_response(data.to_csv())
#         resp.headers["Content-Disposition"] = "attachment; filename=pred.csv"
#         #resp.headers["Content-Type"] = "text/csv"

#         if [(type(col_dtc) == int),(type(col_dts)==int),(type(col_rhob)==int),(type(col_gr)==int)] == [True,True,True,True]:
#             return resp
#         else:
#             return render_template('upload.html', msg='Error: Non-numeric input.')

        
#     return render_template('upload.html')