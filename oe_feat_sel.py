import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SVMSMOTE
from tqdm import tqdm
from online_evolution import OEAlgorithm

if __name__ == '__main__':
    target_map = {'Withdrawn': 0, 'Pass': 1, 'Fail': 2, 'Distinction': 3}
    labels = ['Withdrawn', 'Pass', 'Fail', 'Distinction']
    inverse_map = {0: 'Withdrawn', 1: 'Pass', 2: 'Fail', 3: 'Distinction'}
    inverse_map_func = np.vectorize(lambda x: inverse_map[x])
    
    courses = *map(str, Path("course_stages_data").rglob("*.csv")),
    cols = ['course', 'features', 'accuracy', 'f1', 'precision', 'recall', 'confusion_matrix']
    if Path('./oe_feature_selection.csv').exists():
        results_df = pd.read_csv('./oe_feature_selection.csv', escapechar='\\')
    else:
        results_df = pd.DataFrame(columns=cols)
    for course in tqdm(courses, desc='Courses', unit=' course'):
        course_name = course.split('\\')[-1]
        if course_name in results_df['course'].values:
            continue
        df = pd.read_csv(course)
        data = df.copy()
        data_x = data.drop(['final_result', 'id_student'], axis=1)
        data_y = data['final_result'].map(target_map)
        scaler = StandardScaler()
        data_x_scaled = scaler.fit_transform(data_x)

        model = RandomForestClassifier(random_state=0, n_estimators=70, max_depth=17)
        oe = OEAlgorithm(model, 'accuracy', data_x_scaled[0].shape[0], features_to_select=20)
        oe.fit(data_x_scaled, data_y)

        features_idx = np.where(oe.get_support() == 1)[0]
        data_x_selected = data_x_scaled[:, features_idx]

        data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data_x_selected, data_y, test_size=0.2, random_state=42, stratify=data_y)

        over_sampler = SVMSMOTE(random_state=73)
        data_x_train_sampled, data_y_train_sampled = over_sampler.fit_resample(data_x_train, data_y_train)

        model.fit(data_x_train_sampled, data_y_train_sampled)
        y_pred = model.predict(data_x_test)
        result = course_name, features_idx, accuracy_score(data_y_test, y_pred), f1_score(data_y_test, y_pred, average='weighted'), \
        precision_score(data_y_test, y_pred, average='weighted'), recall_score(data_y_test, y_pred, average='weighted'), \
        confusion_matrix(inverse_map_func(data_y_test), inverse_map_func(y_pred), labels=labels)

        tmp_df = pd.DataFrame([result], columns=cols)
        results_df = pd.concat([results_df, tmp_df], ignore_index=True)
        results_df.to_csv('./oe_feature_selection.csv', index=False)