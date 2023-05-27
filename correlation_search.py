import math

from sqlalchemy import create_engine
import configparser
import pandas as pd
import numpy as np
import scipy as sci

TABLE_NAME = 'test'
a_level = 0.05
p_level = 1 - a_level


def initial_dataframe():
    cfg = configparser.RawConfigParser()
    cfg.read('db.properties')
    engine = create_engine(
        f"postgresql+psycopg2://{cfg.get('DB', 'DBUSER')}:{cfg.get('DB', 'DBPASS')}@localhost:5432/{cfg.get('DB', 'DBNAME')}")
    conn = engine.connect()
    table_name = TABLE_NAME
    data = pd.read_sql("SELECT * FROM " + table_name, conn)
    conn.close()
    return data


def detect_anomalies(raw):
    raw = raw.dropna(axis=0)
    threshold = 3

    mean = np.mean(raw['X'])
    sd = np.std(raw['X'])
    anomX = []
    for i in raw['X']:
        z = (i - mean) / sd  # calculate z-score
        if abs(z) > threshold:  # identify outliers
            anomX.append(i)  # add to the empty list
    for anom in anomX:
        raw.drop(raw[raw['X'] == anom].index, inplace=True)

    mean = np.mean(raw['Y'])
    sd = np.std(raw['Y'])
    anomY = []
    for i in raw['Y']:
        z = (i - mean) / sd  # calculate z-score
        if abs(z) > threshold:  # identify outliers
            anomY.append(i)  # add to the empty list
    for anom in anomY:
        raw.drop(raw[raw['Y'] == anom].index, inplace=True)

    return raw


def Shapiro_Wilk(data):
    data = np.array(data)
    result = sci.stats.shapiro(data)
    a_calc = result.pvalue
    if a_calc >= a_level:
        conclusion_ShW_test = f"Так как a_calc = {round(a_calc, 10)} >= a_level = {round(a_level, 10)}" + \
                              ", то ПРИНИМАЕТСЯ гипотеза о нормальности распределения по критерию Шапиро-Уилка"
    else:
        conclusion_ShW_test = f"Так как a_calc = {round(a_calc, 10)} < a_level = {round(a_level, 10)}" + \
                              ", то ОТВЕРГАЕТСЯ гипотеза о нормальности распределения по критерию Шапиро-Уилка"
    print(conclusion_ShW_test)


def interval_counter(x):
    pred = round(3.31 * math.log(x, 10) + 1)
    if pred >= 2:
        return pred
    else:
        return 2


def Cheddock_scale_check(r, name='r'):
    Cheddock_scale = {
        f'no correlation (|{name}| <= 0.1)': 0.1,
        f'very weak (0.1 < |{name}| <= 0.2)': 0.2,
        f'weak (0.2 < |{name}| <= 0.3)': 0.3,
        f'moderate (0.3 < |{name}| <= 0.5)': 0.5,
        f'perceptible (0.5 < |{name}| <= 0.7)': 0.7,
        f'high (0.7 < |{name}| <= 0.9)': 0.9,
        f'very high (0.9 < |{name}| <= 0.99)': 0.99,
        f'functional (|{name}| > 0.99)': 1.0}

    r_scale = list(Cheddock_scale.values())
    for i, elem in enumerate(r_scale):
        if abs(r) <= elem:
            conclusion_Cheddock_scale = list(Cheddock_scale.keys())[i]
            break
    return conclusion_Cheddock_scale


def Evans_scale_check(r, name='r'):
    Evans_scale = {
        f'very weak (|{name}| < 0.19)': 0.2,
        f'weak (0.2 < |{name}| <= 0.39)': 0.4,
        f'moderate (0.4 < |{name}| <= 0.59)': 0.6,
        f'strong (0.6 < |{name}| <= 0.79)': 0.8,
        f'very strong (0.8 < |{name}| <= 1.0)': 1.0}

    r_scale = list(Evans_scale.values())
    for i, elem in enumerate(r_scale):
        if abs(r) <= elem:
            conclusion_Evans_scale = list(Evans_scale.keys())[i]
            break
    return conclusion_Evans_scale


if __name__ == '__main__':

    #df = initial_dataframe()
    df = pd.read_csv('graduation_rate.csv')

    X = np.array(df['SAT total score'])
    Y = np.array(df['parental income'])

    XY_df = pd.DataFrame({'X': X, 'Y': Y})

    # anomalies detecting
    XY_df = detect_anomalies(XY_df)

    X = np.array(XY_df['X'])
    Y = np.array(XY_df['Y'])

    print(XY_df.describe())
    print("\n")

    Shapiro_Wilk(X)
    Shapiro_Wilk(Y)

    # ----------------------------- 1 -----------------------------------------------

    matrix_XY_df = XY_df.copy()

    # объем выборки для переменных X и Y
    n_X = len(X)
    n_Y = len(Y)

    # число интервалов группировки
    K_X = interval_counter(n_X)
    K_Y = interval_counter(n_Y)
    print(f"\nЧисло интервалов группировки для переменной X: {K_X}")
    print(f"Число интервалов группировки для переменной Y: {K_Y}\n")

    cut_X = pd.cut(X, bins=K_X)
    cut_Y = pd.cut(Y, bins=K_Y)

    matrix_XY_df['cut_X'] = cut_X
    matrix_XY_df['cut_Y'] = cut_Y

    print("Срезы данных:")
    print(matrix_XY_df.head())
    print("\n")

    CorrTable_df = pd.crosstab(
        index=matrix_XY_df['cut_X'],
        columns=matrix_XY_df['cut_Y'],
        rownames=['cut_X'],
        colnames=['cut_Y'])

    print("Корреляционная таблица: \n")
    print(CorrTable_df)

    # проверка правильности подсчета частот по интервалам
    if np.sum(np.array(CorrTable_df)) == len(XY_df):
        print("Значения в корр. таблице совпадают с начальными значениями")
    else:
        print(f"ВНИМАНИЕ! В КОРРЕЛЯЦИОННОЙ ТАБЛИЦЕ НА {len(XY_df - np.sum(np.array(CorrTable_df)))} МЕНЬШЕ ЗНАЧЕНИЙ ЧЕМ В НАЧАЛЬНОЙ ВЫБОРКЕ")

    # ----------------------------- 2 -----------------------------------------------

    CorrTable_np = np.array(CorrTable_df)

    print("Кол-во значений по интервалам")
    n_group_X = [np.sum(CorrTable_np[i]) for i in range(K_X)]
    print(f"n_group_X = {n_group_X}")

    n_group_Y = [np.sum(CorrTable_np[:, j]) for j in range(K_Y)]
    print(f"n_group_Y = {n_group_Y}")

    print("\n")

    print("Среднегрупповые значения")
    Xboun_mean = [(CorrTable_df.index[i].left + CorrTable_df.index[i].right) / 2 for i in range(K_X)]
    Xboun_mean[0] = (np.min(X) + CorrTable_df.index[0].right) / 2
    Xboun_mean[K_X - 1] = (CorrTable_df.index[K_X - 1].left + np.max(X)) / 2
    print(f"Xboun_mean = {Xboun_mean}")

    Yboun_mean = [(CorrTable_df.columns[j].left + CorrTable_df.columns[j].right) / 2 for j in range(K_Y)]
    Yboun_mean[0] = (np.min(Y) + CorrTable_df.columns[0].right) / 2
    Yboun_mean[K_Y - 1] = (CorrTable_df.columns[K_Y - 1].left + np.max(Y)) / 2
    print(f"Yboun_mean = {Yboun_mean}", '\n')

    print("Средневзвешанные значения")
    Xmean_group = [np.sum(CorrTable_np[:, j] * Xboun_mean) / n_group_Y[j] for j in range(K_Y)]
    print(f"Xmean_group = {Xmean_group}")

    Ymean_group = [np.sum(CorrTable_np[i] * Yboun_mean) / n_group_X[i] for i in range(K_X)]
    print(f"Ymean_group = {Ymean_group}")

    print("Дисперсия:")
    disp_total_X = np.sum(n_group_X * (Xboun_mean - np.mean(X)) ** 2)
    print(f"Overall Dispersion X = {disp_total_X}")

    disp_total_Y = np.sum(n_group_Y * (Yboun_mean - np.mean(Y)) ** 2)
    print(f"Overall Dispersion Y = {disp_total_Y}")

    disp_between_X = np.sum(n_group_Y * (Xmean_group - np.mean(X)) ** 2)
    print(f"Between Group Dispersion X = {disp_between_X}")

    disp_between_Y = np.sum(n_group_X * (Ymean_group - np.mean(Y)) ** 2)
    print(f"Between Group Dispersion X = {disp_between_Y}")

    corr_ratio_XY = math.sqrt(disp_between_Y / disp_total_Y)
    print(f"\nCorrelation XY = {corr_ratio_XY}")

    corr_ratio_YX = math.sqrt(disp_between_X / disp_total_X)
    print(f"Correlation YX = {corr_ratio_YX}")

    print(f"Оценка тесноты корреляции (Evans): {Evans_scale_check(corr_ratio_XY, name=chr(951))}")
    print(f"Оценка тесноты корреляции (Cheddock): {Cheddock_scale_check(corr_ratio_XY, name=chr(951))}\n")

    # ------------------------------3-----------------------------------------

    print("Проверка значимости корр. отношения")
    # расчетное значение статистики критерия Фишера
    F_corr_ratio_calc = (n_X - K_X) / (K_X - 1) * corr_ratio_XY ** 2 / (1 - corr_ratio_XY ** 2)
    print(f"Расчетное значение статистики критерия Фишера: F_calc = {round(F_corr_ratio_calc, 10)}")
    # табличное значение статистики критерия Фишера
    dfn = K_X - 1
    dfd = n_X - K_X
    F_corr_ratio_table = sci.stats.f.ppf(p_level, dfn, dfd, loc=0, scale=1)
    print(f"Табличное значение статистики критерия Фишера: F_table = {round(F_corr_ratio_table, 10)}")
    # вывод
    if F_corr_ratio_calc < F_corr_ratio_table:
        conclusion_corr_ratio_sign = f"Так как F_calc < F_table" + \
                                     ", то гипотеза о равенстве нулю корреляционного отношения ПРИНИМАЕТСЯ, т.е. корреляционная связь НЕЗНАЧИМА"
    else:
        conclusion_corr_ratio_sign = f"Так как F_calc >= F_table" + \
                                     ", то гипотеза о равенстве нулю корреляционного отношения ОТВЕРГАЕТСЯ, т.е. корреляционная связь ЗНАЧИМА"
    print(conclusion_corr_ratio_sign)

    # число степеней свободы
    f1 = round((K_X - 1 + n_X * corr_ratio_XY ** 2) ** 2 / (K_X - 1 + 2 * n_X * corr_ratio_XY ** 2))
    f2 = n_X - K_X
    # вспомогательные величины
    z1 = (n_X - K_X) / n_X * corr_ratio_XY ** 2 / (1 - corr_ratio_XY ** 2) * 1 / sci.stats.f.ppf(p_level, f1, f2, loc=0,
                                                                                                 scale=1) - (
                 K_X - 1) / n_X
    z2 = (n_X - K_X) / n_X * corr_ratio_XY ** 2 / (1 - corr_ratio_XY ** 2) * 1 / sci.stats.f.ppf(1 - p_level, f1, f2,
                                                                                                 loc=0, scale=1) - (
                 K_X - 1) / n_X
    # доверительный интервал
    corr_ratio_XY_low = math.sqrt(z1) if math.sqrt(z1) >= 0 else 0
    corr_ratio_XY_high = math.sqrt(z2) if math.sqrt(z2) <= 1 else 1
    print(
        f"\n{p_level * 100}%-ный доверительный интервал для корреляционного отношения: {[round(corr_ratio_XY_low, 10), round(corr_ratio_XY_high, 10)]}")

    # ---------------------------4---------------------------------------

    corr_coef = sci.stats.pearsonr(X, Y)[0]
    print(f"\nКоэффициент линейной корреляции: r = {round(corr_coef, 10)}")

    print(f"Оценка тесноты линейной корреляции (Cheddok): {Cheddock_scale_check(corr_coef)}")
    print(f"Оценка тесноты линейной корреляции (Evans): {Evans_scale_check(corr_coef)}\n")

    # расчетный уровень значимости
    a_corr_coef_calc = sci.stats.pearsonr(X, Y)[1]
    print(f"Расчетный уровень значимости коэффициента линейной корреляции: a_calc = {a_corr_coef_calc}")
    print(f"Заданный уровень значимости: a_level = {round(a_level, 10)}")
    if a_corr_coef_calc >= a_level:
        conclusion_corr_coef_sign = f"Так как a_calc >= a_level" + \
                                    ", то гипотеза о равенстве нулю коэффициента линейной корреляции ПРИНИМАЕТСЯ, т.е. линейная корреляционная связь НЕЗНАЧИМА"
    else:
        conclusion_corr_coef_sign = f"Так как a_calc < a_level" + \
                                    ", то гипотеза о равенстве нулю коэффициента линейной корреляции ОТВЕРГАЕТСЯ, т.е. линейная корреляционная связь ЗНАЧИМА"
    print(conclusion_corr_coef_sign)

    print(f"\nКорреляционное отношение: {chr(951)} = {round(corr_ratio_XY, 10)}")
    print(f"Коэффициент линейной корреляции: r = {round(corr_coef, 10)}")
    # расчетное значение статистики критерия Фишера
    F_line_corr_sign_calc = (n_X - K_X) / (K_X - 2) * (corr_ratio_XY ** 2 - corr_coef ** 2) / (1 - corr_ratio_XY ** 2)
    print(f"Расчетное значение статистики критерия Фишера: F_calc = {round(F_line_corr_sign_calc, 10)}")
    # табличное значение статистики критерия Фишера
    dfn = K_X - 2
    dfd = n_X - K_X
    F_line_corr_sign_table = sci.stats.f.ppf(p_level, dfn, dfd, loc=0, scale=1)
    print(f"Табличное значение статистики критерия Фишера: F_table = {round(F_line_corr_sign_table, 10)}")
    if F_line_corr_sign_calc < F_line_corr_sign_table:
        conclusion_line_corr_sign = f"Так как F_calc < F_table =" + \
                                    f", то гипотеза о равенстве {chr(951)} и r ПРИНИМАЕТСЯ, т.е. корреляционная связь ЛИНЕЙНАЯ"
    else:
        conclusion_line_corr_sign = f"Так как F_calc >= F_table" + \
                                    f", то гипотеза о равенстве {chr(951)} и r ОТВЕРГАЕТСЯ, т.е. корреляционная связь НЕЛИНЕЙНАЯ"
    print(conclusion_line_corr_sign)
