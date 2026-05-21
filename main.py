import numpy as np
import pandas as pd
import gradio as gr
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, 
                           n_redundant=5, n_repeated=2, random_state=42)

feature_names = [f"특성_{i}" for i in range(20)]
df = pd.DataFrame(X, columns=feature_names)
df['특성_0'] = 0.5 

selector_var = VarianceThreshold(threshold=0.01)
X_var = selector_var.fit_transform(df)
features_after_var = df.columns[selector_var.get_support()]

selector_kbest = SelectKBest(score_func=f_classif, k=10)
X_kbest = selector_kbest.fit_transform(X_var, y)
features_after_kbest = features_after_var[selector_kbest.get_support()]

rf_selector = RandomForestClassifier(n_estimators=50, random_state=42)
selector_model = SelectFromModel(rf_selector, threshold="median")
X_final = selector_model.fit_transform(X_kbest, y)
final_features = features_after_kbest[selector_model.get_support()]

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

def predict_interface(*args):
    input_data = np.array(args).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    result = "클래스 1" if prediction == 1 else "클래스 0"
    prob_text = f"확률: 클래스 0 = {probabilities[0]:.2f}, 클래스 1 = {probabilities[1]:.2f}"
    
    return f"예측 결과: {result}\n{prob_text}"

inputs = [
    gr.Number(label=feat, value=0.0) for feat in final_features
]

demo = gr.Interface(
    fn=predict_interface,
    inputs=inputs,
    outputs=gr.Textbox(label="예측 결과"),
    title="특성 선택(Feature Selection) 기반 ML 모델",
    description=f"이 인터페이스는 전체 20개 특성 중 알고리즘을 통해 선택된 {len(final_features)}개의 핵심 특성만 입력받습니다. 모델 정확도: {acc:.2f}"
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)