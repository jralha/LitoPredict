#%%  Checking model on a single unclassified well.
# ############################################################

# new_data = copy.deepcopy(all_dfs)
# new_data = new_data.reset_index()
# new_data = new_data[['wellName','MD','DTc','DTs','RHOB','ECGR']]
# well_list = np.unique(new_data['wellName'].tolist())
# #well_list=['3-BRSA-883-RJS']

# plt.figure(figsize=[15,135])
# for nw,well in enumerate(well_list):
#     print(nw+1)
#     well_name = well
#     print(well_name)

#     plot_data = new_data.loc[new_data['wellName'] == well_name]
#     plot_data['MD'] = plot_data['MD'].astype('float').apply(lambda x: x*-1)
#     feats = plot_data.iloc[:,2:]
#     feat_col = feats.columns
#     feats = imp.fit_transform(feats)
#     feats = pd.DataFrame(feats)
#     feats.columns = feat_col
#     polyfit = PolynomialFeatures().fit(feats)
#     feats = polyfit.fit_transform(feats)

#     wpred = clf0.predict(feats)
#     wprob = clf0.predict_proba(feats)

#     plt.subplot(3,3,nw+1)
    
#     #print('Predictions or probabilisties plot? [pred/prob]')
#     #plot_type = input()
#     plot_type = 'pred'
#     plot_type_set = 0
#     while plot_type_set == 0:
#         if plot_type == 'prob':          
#             plt.plot(wprob.T[1],plot_data['MD'],linewidth=1,color='red')
#             plt.xlabel('Prob BV')
#             plot_type_set=1
#             plt.title(well_name)
#             plt.ylabel('MD')
#             plt.xticks(ticks=[0,1])
#         if plot_type == 'pred':
#             plt.scatter(wpred,plot_data['MD'],s=30,c=wpred,cmap='RdYlGn')
#             plt.xlabel('0 - It, 1 - BV')
#             plot_type_set=1
#             plt.title(well_name)
#             plt.ylabel('MD')
#             plt.xticks(ticks=[0,1])
#         else:
#      #       print('Please input "pred" or "prob".')
#             pass
# plt.tight_layout()