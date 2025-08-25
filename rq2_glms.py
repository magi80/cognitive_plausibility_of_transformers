import pandas as pd
import sys
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.api import abline_plot
from statsmodels.tools.eval_measures import aic
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import numpy as np
from statsmodels.tools.tools import maybe_unwrap_results
from statsmodels.graphics.gofplots import ProbPlot
from scipy import stats


class Padding:

    def __init__(self, write=False):
        self.lang = sys.argv[1]
        self.gpt2_512 = f'CSV_DEBAGGATI/{self.lang}_surprisal_GPT2_512_DEBUG_PREPROCESS_new.csv' # _UPDATE.csv
        self.gpt2_256 = f'CSV_DEBAGGATI/{self.lang}_surprisal_GPT2_256_DEBUG_PREPROCESS_new.csv' # _UPDATE.csv
        self.gpt2_128 = f'CSV_DEBAGGATI/{self.lang}_surprisal_GPT2_128_DEBUG_PREPROCESS_new.csv' # _UPDATE.csv
        self.bert_512 = f'CSV_DEBAGGATI/{self.lang}_surprisal_BERT_512_DEBUG_CW.csv' #BACKUP_CSV
        self.bert_256 = f'CSV_DEBAGGATI/{self.lang}_surprisal_BERT_256_DEBUG_CW.csv'
        self.bert_128 = f'CSV_DEBAGGATI/{self.lang}_surprisal_BERT_128_DEBUG_CW.csv'
        self.all_data = f'{self.lang}_csv_storage/{self.lang}_ALL_DATA-rmd.csv'
        self.chapters = f'{self.lang}_csv_storage/{self.lang}_CHAP-rmd.csv'
        self.write = write


    def open_csv(self):
        all_data = pd.read_csv(self.all_data, delimiter='\t')
        gpt2_512_data = pd.read_csv(self.gpt2_512, delimiter='\t')
        gpt2_256_data = pd.read_csv(self.gpt2_256, delimiter='\t')
        gpt2_128_data = pd.read_csv(self.gpt2_128, delimiter='\t')
        bert_512_data = pd.read_csv(self.bert_512, delimiter='\t')
        bert_256_data = pd.read_csv(self.bert_256, delimiter='\t')
        bert_128_data = pd.read_csv(self.bert_128, delimiter='\t')
        chap_data = pd.read_csv(self.chapters, delimiter='\t')
        return all_data, gpt2_512_data, gpt2_256_data, gpt2_128_data, bert_512_data, bert_256_data, bert_128_data, chap_data
    

    def combine_csv(self, all_data, model_data, chap_data, write=False, log=False):
        #all_data, model_data, chap_data = self.open_csv()
        merged = pd.DataFrame({'tokens': model_data['tokens'],
                               'ar': all_data['ar'],
                               'wl': all_data['wl'],
                               'first': all_data['first'],
                               'last': all_data['last'],
                               'chap': chap_data['chapter'],
                               'surp': model_data['surprisal']}
                               )
        padded_csv = merged.replace('', pd.NA).dropna()
        #print(padded_csv)

        if log:
            padded_csv["log_ar"] = np.log2(padded_csv["ar"]) # log 

        if self.write:
            #padded_csv.to_csv(f'{self.lang}_padded_surprisal_{self.model.upper()}_{self.context_size}_ppp.csv', sep='\t')
            padded_csv.to_csv(f'{self.lang}_padded_surprisal_ppp.csv', sep='\t')
        return padded_csv


    def center_means(self, padded_csv):
        padded_csv['surp_c'] = padded_csv['surp'] - np.mean(padded_csv['surp'])
        padded_csv['wl_c'] = padded_csv['wl'] - np.mean(padded_csv['wl'])
        padded_csv['first_c'] = padded_csv['first'] - np.mean(padded_csv['first'])
        padded_csv['last_c'] = padded_csv['last'] - np.mean(padded_csv['last'])
        return padded_csv


    def get_vif(self, padded_csv):
        X = add_constant(padded_csv[["surp_c", "wl_c", "first_c", "last_c"]])
        X["surp:wl"] = padded_csv["surp_c"] * padded_csv["wl_c"]
        X["surp:first"] = padded_csv["surp_c"] * padded_csv["first_c"]
        X["surp:last"] = padded_csv["surp_c"] * padded_csv["last_c"]
        vifs = pd.DataFrame()
        vifs["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vifs["feature"] = X.columns
        return vifs


    def transform_coeff(self, result):
        coeff = np.exp2(result.params)
        ex_coeff = pd.Series(coeff, index=coeff.index)
        intercept = ex_coeff['Intercept']
        transformed = intercept * (ex_coeff.drop('Intercept') - 1)
        transformed_full = pd.concat([pd.Series({'Intercept': intercept}), transformed])
        return transformed_full


    def glmm(self, padded_csv, model, save_fig=False, write_summary=False):
        padded_csv = self.center_means(padded_csv)
        smodel = smf.glm("log_ar ~ surp_c + wl_c + first_c + last_c + surp_c:first_c + surp_c:last_c + surp_c:wl_c",
                        data=padded_csv, family=sm.families.Gaussian())
        result = smodel.fit() # lbfgs
        res_summary = result.summary()
        
        print('--Transformed:')
        print(self.transform_coeff(result))
        print('-'*20)
        print(res_summary)

        if write_summary:
            cw = model.split('_')[-1]
            path = f'{self.lang}_{model}_{cw}_glmm_summary.txt'
            with open(path, 'w') as f:
                f.write(res_summary.as_text())

            res_aic = aic(llf=result.llf,
                          nobs=result.nobs,
                          df_modelwc=result.df_model)
            
            path_aic = f'{self.lang}_{model}_{cw}_AIC.txt'
            with open(path_aic, 'w') as f:
                f.write(str(res_aic))
        
        print('AIC:', aic(llf=result.llf,
                          nobs=result.nobs,
                          df_modelwc=result.df_model))
        print("Log-Likelihood:", result.llf)
        print("Number of observations:", result.nobs)
        print("Degrees of freedom (model):", result.df_model)
        print("Degrees of freedom (residuals):", result.df_resid)


if __name__ == '__main__':
    model = Padding(write=False)
    all_data, gpt2_512_data, gpt2_256_data, gpt2_128_data, bert_512_data, bert_256_data, bert_128_data, chap_data = model.open_csv() 
    gpt2_512 = model.combine_csv(all_data, gpt2_512_data, chap_data, log=True)
    gpt2_512['model'] = "GPT2-512"
    gpt2_256 = model.combine_csv(all_data, gpt2_256_data, chap_data, log=True)
    gpt2_256['model'] = "GPT2-256"
    gpt2_128 = model.combine_csv(all_data, gpt2_128_data, chap_data, log=True)
    gpt2_128['model'] = "GPT2-128"
    bert_512 = model.combine_csv(all_data, bert_512_data, chap_data, log=True)
    bert_512['model'] = 'BERT-512'
    bert_256 = model.combine_csv(all_data, bert_256_data, chap_data, log=True)
    bert_256['model'] = 'BERT-256'
    bert_128 = model.combine_csv(all_data, bert_128_data, chap_data, log=True)
    bert_128['model'] = 'BERT-128'

    datasets = [(gpt2_512, "GPT2-512"), (bert_512, "BERT-512"), (gpt2_256, "GPT2-256"),
               (bert_256, "BERT-256"), (gpt2_128, "GPT2-128"), (bert_128, "BERT-128")]   
    colors = ["coolwarm", "coolwarm", "coolwarm"]

    for mod in datasets:
        mess = f'# Model {mod[1]} #'
        line = f'# {mess} #'
        print('#'*len(line))
        print(mess)
        print('#'*len(line))
       
        model.glmm(mod[0], mod[1], save_fig=False, write_summary=False)
        print(model.get_vif(mod[0]))
        mod[0].to_csv(f'ENGKJV_padded_prova_{mod[1]}.csv', sep='\t')


    aic_data = {'model': [], 'aic': []}
    for md in datasets:
        smodel = smf.glm("log_ar ~ surp_c + wl_c + first_c + last_c + surp_c:first_c + surp_c:last_c + surp_c:wl_c",
                       data=md[0], family=sm.families.Gaussian())
        nullmodel = smf.glm("log_ar ~ wl_c + first_c + last_c",
                        data=md[0], family=sm.families.Gaussian())
        
        result = smodel.fit() # lbfgs
        result_null = nullmodel.fit()

        delta_ll = result.llf - result_null.llf
        print(result.llf)
        print(result_null.llf)
        print(f"Δ Log-Likelihood for {md[1]}: {delta_ll:.3f}")

        md_aic = aic(llf=result.llf, nobs=result.nobs, df_modelwc=result.df_model)
        aic_data['model'].append(md[1])
        aic_data['aic'].append(md_aic)

    # Sort AIC scores
    aic_df = pd.DataFrame(aic_data)
    aic_df = aic_df.sort_values("aic")
    print(aic_df)

    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=(12, 14), sharey=True, sharex=True) # 18 5
    axes = axes.flatten()  # <- Flatten the 2D array of axes into 1D list

    for ax, (data, label) in zip(axes, datasets):
        sns.scatterplot(
            x="surp_c", y="log_ar", hue='wl', data=data,
            ax=ax, alpha=0.4, s=15, edgecolor='black', palette='coolwarm'
            )
        #sns.regplot(
        #    x="surp_c", y="log_ar", data=data,
        #    scatter=False, ax=ax
        
        sns.lmplot(
            x="surp_c", y="log_ar", hue="wl_c", data=data,
            palette="coolwarm", height=4, aspect=1.5, scatter_kws={"s": 10, "alpha": 0.3}
            )

        ax.set_title(label)
        ax.set_xlabel("Surprisal")
        ax.set_ylabel("AR (syllable/sec) - Log2")

    plt.suptitle("GLMs: Effect of Surprisal on Log Articulation Rate by Model Size", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    #plt.savefig('GLMS.png', dpi=300)


    