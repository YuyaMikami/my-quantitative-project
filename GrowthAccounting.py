import pandas as pd
import numpy as np

# PWT 9.0 データ読み込み
pwt90 = pd.read_stata('https://www.rug.nl/ggdc/docs/pwt90.dta')

# OECD国名修正済み
oecd_countries = [
    'Australia','Austria','Belgium','Canada','Denmark','Finland','France',
    'Germany','Greece','Iceland','Ireland','Italy','Japan','Netherlands',
    'New Zealand','Norway','Portugal','Spain','Sweden','Switzerland',
    'United Kingdom','United States'
]

# 1990〜2019年のデータ抽出
data = pwt90[
    pwt90['country'].isin(oecd_countries) &
    pwt90['year'].between(1990, 2019)
]

# 必要な列のみ抽出
relevant_cols = ['countrycode', 'country', 'year', 'rgdpna', 'rkna', 'emp', 'labsh']
data = data[relevant_cols].dropna()

# 一人当たり GDP と 資本装備率 を計算
data['y_n'] = data['rgdpna'] / data['emp']         # Y/N
data['k_n'] = data['rkna'] / data['emp']           # K/N
data['alpha'] = 1 - data['labsh']                  # α = 1 - 労働所得比率

# 年次成長率（log差分）
data = data.sort_values(['country', 'year'])
data['g_y'] = data.groupby('country')['y_n'].transform(lambda x: np.log(x).diff())
data['g_k'] = data.groupby('country')['k_n'].transform(lambda x: np.log(x).diff())
data['alpha'] = data.groupby('country')['alpha'].transform(lambda x: x.fillna(method='ffill'))

# 成長率分解：TFP = Y成長 - α × K/N成長
data['capital_deepening'] = data['alpha'] * data['g_k']
data['tfp_growth'] = data['g_y'] - data['capital_deepening']

# 平均値の計算
results = data.groupby('country').agg({
    'g_y': 'mean',
    'capital_deepening': 'mean',
    'tfp_growth': 'mean'
}).dropna().reset_index()

# パーセントに直す
results['Growth Rate'] = results['g_y'] * 100
results['Capital Deepening'] = results['capital_deepening'] * 100
results['TFP Growth'] = results['tfp_growth'] * 100

# シェア（割合）
results['TFP Share'] = results['TFP Growth'] / results['Growth Rate']
results['Capital Share'] = results['Capital Deepening'] / results['Growth Rate']

# 平均行を追加
avg_row = {
    'country': 'Average',
    'Growth Rate': results['Growth Rate'].mean(),
    'Capital Deepening': results['Capital Deepening'].mean(),
    'TFP Growth': results['TFP Growth'].mean(),
    'TFP Share': results['TFP Share'].mean(),
    'Capital Share': results['Capital Share'].mean()
}
results = pd.concat([results, pd.DataFrame([avg_row])], ignore_index=True)

# 出力
print("\nGrowth Accounting (Y/N) in OECD Countries: 1990–2019")
print("=" * 90)
print(results[['country', 'Growth Rate', 'TFP Growth', 'Capital Deepening', 'TFP Share', 'Capital Share']].to_string(index=False, float_format="%.2f"))
