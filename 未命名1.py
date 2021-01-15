    def Predict(self, min_or_more = 'min', pred_options = {'is_total':False, 'combin_func':'avg'}):
        '''读取测试数据，且各个准则进行预测'''
        names = locals()
        r = Reader(rating_scale=(1,5))
        # if min_or_more == 'min':
        #     df = self.min_test.sort_values(by='uid')
        # else:
        #     df = self.more_test.sort_values(by='uid')
        df = self.testDatas
        total_test = np.array(df[['uid','iid','total']])
        total_p = self.algos[0].test(total_test)
        for i in range(1, self.no_of_criteria+1):
            # names['c' + str(i) + '_test'] = np.array(df[['uid','iid', 'c' + str(i)]])
            names['c' + str(i) + '_test'] = np.array(df[['uid','iid', 'c' + str(i)]])
            names['c' + str(i) + '_p'] = self.algos[i].test(names.get('c' + str(i) + '_test'))
        
        multi_p = []
        if pred_options['is_total']:
            if pred_options['combin_func'] == 'avg':
                for i in range(len(total_p)):
                    s = 0
                    for j in range(1, self.no_of_criteria+1):
                        s = s + names.get('c'+str(j)+'_p')[i].est
                    avg = (s + total_p[i].est) / (self.no_of_criteria + 1)
                    p = predictions.Prediction(total_p[i].uid, total_p[i].iid, total_p[i].r_ui, avg, total_p[i].details)
                    multi_p.append(p)
            else:
                print('总分作为准则不适合用于回归聚合函数')
        else:
            if pred_options['combin_func'] == 'avg':
                for i in range(len(total_p)):
                    s = 0
                    for j in range(1, self.no_of_criteria+1):
                        s = s + names.get('c'+str(j)+'_p')[i].est
                    avg = s / self.no_of_criteria
                    p = predictions.Prediction(total_p[i].uid, total_p[i].iid, total_p[i].r_ui, avg, total_p[i].details)
                    multi_p.append(p)
            elif pred_options['combin_func'] == 'total_reg':
                k = self.k
                b = self.b
                for i in range(len(total_p)):
                    s = 0
                    for j in range(self.no_of_criteria):
                        s = s + k[j]*names.get('c'+str(j+1)+'_p')[i].est
                    s = s + b
                    p = predictions.Prediction(total_p[i].uid, total_p[i].iid, total_p[i].r_ui, s, total_p[i].details)
                    multi_p.append(p)
            elif pred_options['combin_func'] == 'info_entropy':
                self.Info_Entropy()
                H = np.array(self.H)
                
                for i in range(len(total_p)):
                    s = 0
                    if len(np.argwhere(H[:,0] == total_p[i].uid)):
                        h = H[np.argwhere(H[:,0] == total_p[i].uid)][0][0]
                        for j in range(1, self.no_of_criteria+1):
                            s = s + h[j]*names.get('c'+str(j)+'_p')[i].est/h[self.no_of_criteria+1]
                        
                        p = predictions.Prediction(total_p[i].uid, total_p[i].iid, total_p[i].r_ui, s, total_p[i].details)
                        multi_p.append(p)
                    else:
                        s = 0
                        for j in range(1, self.no_of_criteria+1):
                            s = s + names.get('c'+str(j)+'_p')[i].est
                        s = s/self.no_of_criteria
                        p = predictions.Prediction(total_p[i].uid, total_p[i].iid, total_p[i].r_ui, s, total_p[i].details)
                        multi_p.append(p)


                        
                        
                               
        s_mae = round(accuracy.mae(total_p),4)     
        m_mae = round(accuracy.mae(multi_p),4)        
        return s_mae, m_mae