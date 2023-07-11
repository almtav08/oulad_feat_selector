import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFECV, SequentialFeatureSelector, SelectFromModel
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import kruskal
from collections import defaultdict
from imblearn.over_sampling import SVMSMOTE
from tqdm import tqdm
from func_timeout import func_timeout

KMIC = {'score_func': 'mutual_info_classif', 'k': 20}
KKW = {'score_func': 'kruskal_wallis', 'k': 20}
RFECVE = {'estimator': SVC(random_state=0, probability=True, kernel='linear', C=0.9, degree=1), 'cv': 10, 'min_features_to_select': 20, 'step': 0.05, 'n_jobs': 14}
SFS = {'estimator': SVC(random_state=0, probability=True, kernel='linear', C=0.9, degree=1), 'cv': 10, 'n_features_to_select': 20, 'n_jobs': 14}
SFM = {'estimator': RandomForestClassifier(random_state=0, n_estimators=70, max_depth=17), 'max_features': 20}
SELECTORS = {'KMIC': 'SelectKBest', 'KKW': 'SelectKBest', 'RFECVE': 'RFECV', 'SFS': 'SequentialFeatureSelector', 'SFM': 'SelectFromModel'}

def kruskal_wallis(X, y):
    k_values = []
    p_values = []
    for feature in X.T:
        stat, p = kruskal(*[feature[y == cls] for cls in np.unique(y)])
        k_values.append(stat)
        p_values.append(p)
    return np.array(k_values), np.array(p_values)


def vote_features(course: str, course_data: pd.DataFrame, target_map: dict, labels: list, inverse_map_func: np.vectorize):
    data = course_data.copy()
    data_x = data.drop(['final_result', 'id_student'], axis=1)
    data_y = data['final_result'].map(target_map)
    scaler = StandardScaler()
    data_x_scaled = scaler.fit_transform(data_x)

    features_dict = defaultdict(int)
    for sel in SELECTORS:
        conf = eval(sel)
        if sel == 'KMIC':
            selector = SelectKBest(score_func=eval(conf['score_func']), k=conf['k'])
        elif sel == 'KKW':
            selector = SelectKBest(score_func=kruskal_wallis, k=conf['k'])
        elif sel == 'RFECVE' or sel == 'SFS' or sel == 'SFM':
            conf['estimator'] = clone(conf['estimator'])
            selector = eval(f'{SELECTORS[sel]}(**{conf})')
        try:
            func_timeout(2500, selector.fit, args=(data_x_scaled, data_y))
        except:
            pass
        if (sel in ['KMIC', 'KKW', 'SFM']) or (sel in ['RFECVE', 'SFS']  and hasattr(selector, 'support_')):
            print(f'{sel} - {selector.get_support().sum()}')
            for feat in selector.get_feature_names_out():
                features_dict[feat] += 1

    features = sorted(features_dict, key=lambda x: -features_dict[x])[:20]
    features_idx = [int(feat[1:]) for feat in features]
    data_x_selected = data_x_scaled[:, features_idx]
    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data_x_selected, data_y, test_size=0.2, random_state=42, stratify=data_y)
    over_sampler = SVMSMOTE(random_state=73)
    data_x_train_sampled, data_y_train_sampled = over_sampler.fit_resample(data_x_train, data_y_train)

    model = RandomForestClassifier(random_state=0, n_estimators=70, max_depth=17)
    model.fit(data_x_train_sampled, data_y_train_sampled)
    y_pred = model.predict(data_x_test)

    return course, features_idx, accuracy_score(data_y_test, y_pred), f1_score(data_y_test, y_pred, average='weighted'), \
        precision_score(data_y_test, y_pred, average='weighted'), recall_score(data_y_test, y_pred, average='weighted'), \
        confusion_matrix(inverse_map_func(data_y_test), inverse_map_func(y_pred), labels=labels)


if __name__ == '__main__':
    target_map = {'Withdrawn': 0, 'Pass': 1, 'Fail': 2, 'Distinction': 3}
    labels = ['Withdrawn', 'Pass', 'Fail', 'Distinction']
    inverse_map = {0: 'Withdrawn', 1: 'Pass', 2: 'Fail', 3: 'Distinction'}
    inverse_map_func = np.vectorize(lambda x: inverse_map[x])
    
    courses = *map(str, Path("course_stages_data").rglob("*.csv")),
    cols = ['course', 'features', 'accuracy', 'f1', 'precision', 'recall', 'confusion_matrix']
    if Path('./feature_selection.csv').exists():
        results_df = pd.read_csv('./feature_selection.csv')
    else:
        results_df = pd.DataFrame(columns=cols)
    completed_courses = results_df['course'].values
    remain_courses = [course for course in courses if course not in completed_courses]
    for course in tqdm(remain_courses, desc='Courses', unit=' course'):
        print(f"Processing {course}")
        course_data = pd.read_csv(course)
        result = vote_features(course, course_data, target_map, labels, inverse_map_func)
        # tmp_df = pd.DataFrame([result], columns=cols)
        # results_df = pd.concat([results_df, tmp_df], ignore_index=True)
        # results_df.to_csv('./feature_selection.csv', index=False)
