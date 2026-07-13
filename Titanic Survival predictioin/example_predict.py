"""Simple example showing how to use titanic_model_interface.TitanicModelInterface
Run:
    python example_predict.py C:\path\to\model.pkl
"""

import sys
from titanic_model_interface import TitanicModelInterface


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    iface = TitanicModelInterface()
    if model_path:
        try:
            iface.load_model(model_path)
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            return
    else:
        print("No model path provided. Pass path to sklearn joblib/pickle or Keras .h5 file as first argument.")
        return

    sample = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S",
    }
    res = iface.predict(sample)
    print("Input:", sample)
    print("Prediction:", res)


if __name__ == "__main__":
    main()
