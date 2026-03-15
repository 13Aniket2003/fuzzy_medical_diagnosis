import os
import subprocess
from time import sleep
import sys
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,recall_score, f1_score, confusion_matrix)
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --------- Paths & dataset info ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
os.makedirs(DATA_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "pima.csv")   # raw
PROCESSED_PATH = os.path.join(DATA_DIR, "pima_processed.csv")
PRED_PATH = os.path.join(DATA_DIR, "predictions.csv")
HIST_PATH = os.path.join(DATA_DIR, "fuzzy_scores_hist.png")
CM_FUZZY_PATH = os.path.join(DATA_DIR, "confusion_fuzzy.png")
CM_LR_PATH = os.path.join(DATA_DIR, "confusion_lr.png")

# # --------- preprocess ----------

def load_and_preprocess():
    cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin","BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    df = pd.read_csv(CSV_PATH, header=None, names=cols)

    # Replace zeros with the median value
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for c in zero_as_missing:
        s = df[c].replace(0, np.nan)
        med = s.median()
        df.loc[:, c] = s.fillna(med)

    df.to_csv(PROCESSED_PATH, index=False)
    print("Preprocessing complete. Saved:", PROCESSED_PATH)
    return df

# --------- Fuzzy system builder ----------
def build_fuzzy_system_and_meta():

    glucose = ctrl.Antecedent(np.arange(0, 201, 1), 'glucose')
    bmi = ctrl.Antecedent(np.arange(10, 61, 0.1), 'bmi')
    bp = ctrl.Antecedent(np.arange(0, 181, 1), 'bp')
    age = ctrl.Antecedent(np.arange(10, 101, 1), 'age')
    risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

    # Membership functions 
    glucose['low'] = fuzz.trapmf(glucose.universe, [0,0,70,90])
    glucose['normal'] = fuzz.trimf(glucose.universe, [80,110,140])
    glucose['high'] = fuzz.trapmf(glucose.universe, [120,140,200,200])

    bmi['low'] = fuzz.trapmf(bmi.universe, [10,10,16,18.5])
    bmi['normal'] = fuzz.trimf(bmi.universe, [18,24,30])
    bmi['high'] = fuzz.trapmf(bmi.universe, [28,30,60,60])

    bp['low'] = fuzz.trapmf(bp.universe, [0,0,60,70])
    bp['normal'] = fuzz.trimf(bp.universe, [65,80,90])
    bp['high'] = fuzz.trapmf(bp.universe, [85,95,180,180])

    age['young'] = fuzz.trapmf(age.universe, [10,10,25,35])
    age['middle'] = fuzz.trimf(age.universe, [30,45,60])
    age['old'] = fuzz.trapmf(age.universe, [55,65,100,100])

    risk['low'] = fuzz.trapmf(risk.universe, [0,0,20,40])
    risk['medium'] = fuzz.trimf(risk.universe, [30,50,70])
    risk['high'] = fuzz.trapmf(risk.universe, [60,80,100,100])

    antecedents = {'glucose': glucose, 'bmi': bmi, 'bp': bp, 'age': age}
    rules = []
    rules_meta = []

    def add_rule(ante_pairs, consequent_term):
        expr = None
        parts = []
        for var, term in ante_pairs:
            part = antecedents[var][term]
            parts.append(f"{var}['{term}']")
            expr = part if expr is None else (expr & part)
        rule_obj = ctrl.Rule(expr, risk[consequent_term])
        rules.append(rule_obj)
        desc = " AND ".join(parts)
        rules_meta.append((rule_obj, ante_pairs, desc))
        return rule_obj

    # combined rules
    add_rule([('glucose','high'), ('bmi','high')], 'high')
    add_rule([('glucose','high'), ('age','old')], 'high')
    add_rule([('glucose','high'), ('bp','high')], 'high')
    add_rule([('glucose','normal'), ('bmi','normal'), ('bp','normal')], 'low')
    add_rule([('glucose','low'), ('bmi','low')], 'low')
    add_rule([('age','old'), ('bmi','high')], 'medium')
    add_rule([('bp','high'), ('bmi','high')], 'medium')

    # Single-term rules
    add_rule([('glucose','low')], 'low')
    add_rule([('bmi','low')], 'low')
    add_rule([('bp','low')], 'low')
    add_rule([('age','young')], 'low')

    add_rule([('glucose','normal')], 'medium')
    add_rule([('bmi','normal')], 'medium')
    add_rule([('age','middle')], 'medium')

    add_rule([('glucose','high')], 'high')
    add_rule([('bmi','high')], 'high')
    add_rule([('bp','high')], 'high')

    # Additional rules
    add_rule([('glucose','normal'), ('bmi','high')], 'medium')
    add_rule([('glucose','low'), ('age','young')], 'low')

    system = ctrl.ControlSystem(rules)
    return system, antecedents, rules_meta

# --------- Fallback heuristic ----------
def fallback_heuristic(ante_dict, input_map):
    try:
        gh = fuzz.interp_membership(ante_dict['glucose'].universe, ante_dict['glucose']['high'].mf, input_map['glucose'])
        bh = fuzz.interp_membership(ante_dict['bmi'].universe, ante_dict['bmi']['high'].mf, input_map['bmi'])
        bph = fuzz.interp_membership(ante_dict['bp'].universe, ante_dict['bp']['high'].mf, input_map['bp'])
        ao = fuzz.interp_membership(ante_dict['age'].universe, ante_dict['age']['old'].mf, input_map['age'])
        score = 0.5*gh + 0.3*bh + 0.1*bph + 0.1*ao
        return float(score * 100)
    except Exception:
        return 0.0

# --------- Rule strength helper ----------
def compute_rule_strengths(rules_meta, ante_dict, input_map):
    strengths = []
    for (_rule, ante_pairs, desc) in rules_meta:
        degs = []
        for var, term in ante_pairs:
            try:
                deg = fuzz.interp_membership(ante_dict[var].universe, ante_dict[var][term].mf, input_map[var])
                degs.append(float(deg))
            except Exception:
                degs.append(0.0)
        strength = min(degs) if degs else 0.0
        strengths.append((desc, strength))
    strengths.sort(key=lambda x: x[1], reverse=True)
    return strengths

# --------- Metrics helper ----------
def print_metrics(y_true, y_pred, label="Model"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n{label}: ACC={acc:.3f} PREC={prec:.3f} REC={rec:.3f} F1={f1:.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}

# --------- Main evaluation pipeline ----------
def evaluate_and_save():
    df = load_and_preprocess()

    features = ["Glucose", "BMI", "BloodPressure", "Age"]
    X = df[features].copy()
    y = df["Outcome"].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42, stratify=y)

    system, ante_dict, rules_meta = build_fuzzy_system_and_meta()

    y_pred_fuzzy = []
    fuzzy_scores = []
    top_rule_descs = []
    top_rule_strengths = []

    for idx, row in X_test.iterrows():
        sim = ctrl.ControlSystemSimulation(system)
        sim.input['glucose'] = float(row['Glucose'])
        sim.input['bmi'] = float(row['BMI'])
        sim.input['bp'] = float(row['BloodPressure'])
        sim.input['age'] = float(row['Age'])

        try:
            sim.compute()
            if 'risk' in sim.output:
                out = float(sim.output['risk'])
                used_fallback = False
            else:
                out = fallback_heuristic(ante_dict, {'glucose': float(row['Glucose']),
                                                     'bmi': float(row['BMI']),
                                                     'bp': float(row['BloodPressure']),
                                                      'age': float(row['Age'])})
                used_fallback = True
        except Exception:
            out = fallback_heuristic(ante_dict, {'glucose': float(row['Glucose']),
                                                  'bmi': float(row['BMI']),
                                                  'bp': float(row['BloodPressure']),
                                                  'age': float(row['Age'])})
            used_fallback = True

        fuzzy_scores.append(out)
        y_pred_fuzzy.append(1 if out >= 50 else 0)

        # compute top rule
        input_map = {'glucose': float(row['Glucose']), 'bmi': float(row['BMI']),
                     'bp': float(row['BloodPressure']), 'age': float(row['Age'])}
        strengths = compute_rule_strengths(rules_meta, ante_dict, input_map)
        top_desc, top_strength = (strengths[0] if strengths else ("<no rule>", 0.0))
        top_rule_descs.append(top_desc)
        top_rule_strengths.append(top_strength)

    print_metrics(y_test, y_pred_fuzzy, label="Fuzzy System")

    # Baseline logistic regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print_metrics(y_test, y_pred_lr, label="Logistic Regression")

    # predictions
    out_df = X_test.copy()
    out_df.reset_index(inplace=True, drop=True)
    out_df['y_true'] = y_test.values
    out_df['risk_score'] = fuzzy_scores
    out_df['pred_fuzzy'] = y_pred_fuzzy
    out_df['top_rule'] = top_rule_descs
    out_df['top_rule_strength'] = top_rule_strengths
    out_df['pred_lr'] = y_pred_lr
    out_df.to_csv(PRED_PATH, index=False)
    print("Saved predictions to:", PRED_PATH)

    # histogram and confusion matrices

    def try_open_file(path):
        try:
            if os.name == "nt":  # Windows
                os.startfile(path)
            elif sys.platform == "darwin":  # macOS
                subprocess.call(["open", path])
            else:  # Linux and others
                subprocess.call(["xdg-open", path])
        except Exception as e:
            print("Couldn't open file with OS viewer:", e)

    # 1) Histogram
    try:
        fig = plt.figure(figsize=(8,4))
        plt.hist(fuzzy_scores, bins=20)
        plt.title("Distribution of fuzzy risk scores (test set)")
        plt.xlabel("Risk (0-100)")
        plt.ylabel("Count")
        fig.savefig(HIST_PATH, bbox_inches='tight', dpi=150)
        print("Saved histogram to:", HIST_PATH)

        # show using matplotlib
        try:
            plt.show(block=True)
        except Exception as e_show:
            print("plt.show() failed for histogram (will try OS viewer):", e_show)
            try_open_file(HIST_PATH)
        finally:
            plt.close(fig)
            sleep(0.3)  # small pause to allow viewer to open
    except Exception as e:
        print("Could not save/show histogram:", e)

    # 2) Fuzzy confusion matrix 
    try:
        cm_f = confusion_matrix(y_test, y_pred_fuzzy)
        fig, ax = plt.subplots(figsize=(4,3))
        ax.imshow(cm_f, interpolation='nearest', cmap='Blues')
        for i in range(cm_f.shape[0]):
            for j in range(cm_f.shape[1]):
                ax.text(j, i, str(cm_f[i, j]), ha="center", va="center", color="Black")
        ax.set_title("Confusion matrix (Fuzzy)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.savefig(CM_FUZZY_PATH, bbox_inches='tight', dpi=150)
        print("Saved fuzzy confusion matrix to:", CM_FUZZY_PATH)

        try:
            plt.show(block=True)
        except Exception as e_show:
            print("plt.show() failed for fuzzy confusion (will try OS viewer):", e_show)
            try_open_file(CM_FUZZY_PATH)
        finally:
            plt.close(fig)
            sleep(0.3)
    except Exception as e:
        print("Could not save/show fuzzy confusion matrix:", e)

    # 3) Logistic Regression confusion matrix 
    try:
        cm_lr = confusion_matrix(y_test, y_pred_lr)
        fig, ax = plt.subplots(figsize=(4,3))
        ax.imshow(cm_lr, interpolation='nearest', cmap='Greens')
        for i in range(cm_lr.shape[0]):
            for j in range(cm_lr.shape[1]):
                ax.text(j, i, str(cm_lr[i, j]), ha="center", va="center", color="black")
        ax.set_title("Confusion matrix (Logistic Regression)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.savefig(CM_LR_PATH, bbox_inches='tight', dpi=150)
        print("Saved LR confusion matrix to:", CM_LR_PATH)

        try:
            plt.show(block=True)
        except Exception as e_show:
            print("plt.show() failed for LR confusion (will try OS viewer):", e_show)
            try_open_file(CM_LR_PATH)
        finally:
            plt.close(fig)
            sleep(0.3)
    except Exception as e:
        print("Could not save/show LR confusion matrix:", e)

# Run
if __name__ == "__main__":
    evaluate_and_save()