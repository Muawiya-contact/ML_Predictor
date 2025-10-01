{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "41837ef8-35fc-442c-8892-500a2595bd8e",
   "metadata": {},
   "source": [
    "# Step 1: Import Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c895b656-6a2d-4d52-85a7-bcf8920e99b7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score, classification_report"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4372fcc3-d764-46fc-9269-5d877e203739",
   "metadata": {},
   "source": [
    "# Step 2: Load Your Dataset\n",
    " File `CardiacPatientData.csv`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "b0f1b528-7482-4e1e-9636-6a24857a4937",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>SBP</th>\n",
       "      <th>DBP</th>\n",
       "      <th>HR</th>\n",
       "      <th>RR</th>\n",
       "      <th>BT</th>\n",
       "      <th>SpO2</th>\n",
       "      <th>Age</th>\n",
       "      <th>Gender</th>\n",
       "      <th>GCS</th>\n",
       "      <th>Na</th>\n",
       "      <th>K</th>\n",
       "      <th>Cl</th>\n",
       "      <th>Urea</th>\n",
       "      <th>Ceratinine</th>\n",
       "      <th>Alcoholic</th>\n",
       "      <th>Smoke</th>\n",
       "      <th>FHCD</th>\n",
       "      <th>TriageScore</th>\n",
       "      <th>Outcome</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>163</td>\n",
       "      <td>95</td>\n",
       "      <td>90</td>\n",
       "      <td>18</td>\n",
       "      <td>98</td>\n",
       "      <td>98</td>\n",
       "      <td>66</td>\n",
       "      <td>1</td>\n",
       "      <td>15</td>\n",
       "      <td>139.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>105.0</td>\n",
       "      <td>41.0</td>\n",
       "      <td>91.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>134</td>\n",
       "      <td>85</td>\n",
       "      <td>85</td>\n",
       "      <td>15</td>\n",
       "      <td>98</td>\n",
       "      <td>98</td>\n",
       "      <td>66</td>\n",
       "      <td>1</td>\n",
       "      <td>15</td>\n",
       "      <td>139.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>105.0</td>\n",
       "      <td>41.0</td>\n",
       "      <td>91.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>121</td>\n",
       "      <td>77</td>\n",
       "      <td>80</td>\n",
       "      <td>19</td>\n",
       "      <td>98</td>\n",
       "      <td>98</td>\n",
       "      <td>66</td>\n",
       "      <td>1</td>\n",
       "      <td>15</td>\n",
       "      <td>139.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>105.0</td>\n",
       "      <td>41.0</td>\n",
       "      <td>91.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>103</td>\n",
       "      <td>78</td>\n",
       "      <td>70</td>\n",
       "      <td>16</td>\n",
       "      <td>98</td>\n",
       "      <td>98</td>\n",
       "      <td>66</td>\n",
       "      <td>1</td>\n",
       "      <td>15</td>\n",
       "      <td>139.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>105.0</td>\n",
       "      <td>41.0</td>\n",
       "      <td>91.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>96</td>\n",
       "      <td>70</td>\n",
       "      <td>59</td>\n",
       "      <td>13</td>\n",
       "      <td>98</td>\n",
       "      <td>98</td>\n",
       "      <td>66</td>\n",
       "      <td>1</td>\n",
       "      <td>15</td>\n",
       "      <td>139.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>105.0</td>\n",
       "      <td>41.0</td>\n",
       "      <td>91.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   ID  SBP  DBP  HR  RR  BT  SpO2  Age  Gender  GCS     Na    K     Cl  Urea  \\\n",
       "0   1  163   95  90  18  98    98   66       1   15  139.0  4.0  105.0  41.0   \n",
       "1   1  134   85  85  15  98    98   66       1   15  139.0  4.0  105.0  41.0   \n",
       "2   1  121   77  80  19  98    98   66       1   15  139.0  4.0  105.0  41.0   \n",
       "3   1  103   78  70  16  98    98   66       1   15  139.0  4.0  105.0  41.0   \n",
       "4   1   96   70  59  13  98    98   66       1   15  139.0  4.0  105.0  41.0   \n",
       "\n",
       "   Ceratinine  Alcoholic  Smoke  FHCD  TriageScore  Outcome  \n",
       "0        91.0        1.0    1.0   0.0          3.0        1  \n",
       "1        91.0        1.0    1.0   0.0          3.0        1  \n",
       "2        91.0        1.0    1.0   0.0          3.0        1  \n",
       "3        91.0        1.0    1.0   0.0          3.0        1  \n",
       "4        91.0        1.0    1.0   0.0          3.0        1  "
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv(\"CardiacPatientData.csv\")  # example: heart disease dataset\n",
    "df.head()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "655b7089-82bd-4b69-966e-57a3509f3e99",
   "metadata": {},
   "source": [
    "# Step 3: Prepare Features (X) and Target (y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "a725fae0-07d6-4891-8fed-dbac628dcae7",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.drop(columns=[\"ID\", \"Outcome\"])\n",
    "y = df[\"Outcome\"]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "af85429c-277f-4e23-b64d-80d36b1b5f65",
   "metadata": {},
   "source": [
    "# Step 4: Train-Test Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "bd3c74fa-9f0d-4005-9f77-babedda66583",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=42)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fe6a4bf5-cc3b-4537-b64b-21bf1adda8dc",
   "metadata": {},
   "source": [
    "# Step 5: Train the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "ad70a4d4-eb0f-4504-8081-76196129a0cd",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "model.fit(X_train, y_train);\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6fb6eb0a-884f-419f-a18e-1ed91534b270",
   "metadata": {},
   "source": [
    "#  Step 6: Evaluate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "e90a899f-f4cc-4a15-b68a-f78409124315",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1507    1\n",
       "5569    0\n",
       "5206    0\n",
       "3469    1\n",
       "3839    1\n",
       "       ..\n",
       "4934    1\n",
       "3968    1\n",
       "1097    1\n",
       "3560    1\n",
       "381     1\n",
       "Name: Outcome, Length: 591, dtype: int64"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "57551294-eb9a-4a64-9824-7b1ef4f8ad2e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.9915397631133672\n",
      "\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      0.94      0.97        82\n",
      "           1       0.99      1.00      1.00       509\n",
      "\n",
      "    accuracy                           0.99       591\n",
      "   macro avg       1.00      0.97      0.98       591\n",
      "weighted avg       0.99      0.99      0.99       591\n",
      "\n"
     ]
    }
   ],
   "source": [
    "y_pred = model.predict(X_test)\n",
    "\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred))\n",
    "print(\"\\nClassification Report:\\n\", classification_report(y_test, y_pred))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "8013c6bb-99ba-4b87-b1af-ec6b019762fb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0,\n",
       "       1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1,\n",
       "       1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1,\n",
       "       1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0,\n",
       "       1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1,\n",
       "       0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1,\n",
       "       0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1,\n",
       "       1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1,\n",
       "       0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,\n",
       "       1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "990fa189-9eac-46eb-9727-c92c6a8fcf58",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predicted Outcome: [0]\n"
     ]
    }
   ],
   "source": [
    "# Take the first row from X_test as a DataFrame\n",
    "sample_df = X_test.iloc[[2]]  # note the double brackets to keep it as DataFrame\n",
    "\n",
    "# Predict\n",
    "prediction = model.predict(sample_df)\n",
    "print(\"Predicted Outcome:\", prediction)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "511db75b-239c-483f-9868-ea9153cceb5c",
   "metadata": {},
   "source": [
    "# The End."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "1b773820-6a4d-4f35-ae79-96b7eccf0793",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 5 folds for each of 216 candidates, totalling 1080 fits\n",
      "Best Parameters: {'bootstrap': False, 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}\n",
      "Best Score: 0.9920978363123236\n",
      "Test Accuracy: 0.9915397631133672\n"
     ]
    }
   ],
   "source": [
    "\n",
    "param_grid = {\n",
    "    'n_estimators': [100, 200, 300],       # number of trees\n",
    "    'max_depth': [None, 5, 10, 20],        # tree depth\n",
    "    'min_samples_split': [2, 5, 10],       # min samples to split a node\n",
    "    'min_samples_leaf': [1, 2, 4],         # min samples per leaf\n",
    "    'bootstrap': [True, False]             # use bootstrap sampling\n",
    "}\n",
    "grid_search = GridSearchCV(\n",
    "    estimator=RandomForestClassifier(random_state=42),\n",
    "    param_grid=param_grid,\n",
    "    cv=5,                   # 5-fold cross-validation\n",
    "    scoring='accuracy',\n",
    "    n_jobs=-1,              # use all CPU cores\n",
    "    verbose=2\n",
    ")\n",
    "\n",
    "grid_search.fit(X_train, y_train)\n",
    "print(\"Best Parameters:\", grid_search.best_params_)\n",
    "print(\"Best Score:\", grid_search.best_score_)\n",
    "best_model = grid_search.best_estimator_\n",
    "\n",
    "# Evaluate on test set\n",
    "y_pred = best_model.predict(X_test)\n",
    "print(\"Test Accuracy:\", accuracy_score(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "7b620844-7df2-4f6b-a838-dba048efdcee",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'best_model' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[31m---------------------------------------------------------------------------\u001b[39m",
      "\u001b[31mNameError\u001b[39m                                 Traceback (most recent call last)",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[10]\u001b[39m\u001b[32m, line 1\u001b[39m\n\u001b[32m----> \u001b[39m\u001b[32m1\u001b[39m y_pred = \u001b[43mbest_model\u001b[49m.predict(X_test)\n\u001b[32m      2\u001b[39m \u001b[38;5;28mprint\u001b[39m(\u001b[33m\"\u001b[39m\u001b[33mTest Accuracy:\u001b[39m\u001b[33m\"\u001b[39m, accuracy_score(y_test, y_pred))\n",
      "\u001b[31mNameError\u001b[39m: name 'best_model' is not defined"
     ]
    }
   ],
   "source": [
    "y_pred = best_model.predict(X_test)\n",
    "print(\"Test Accuracy:\", accuracy_score(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "8d71f497-f9fa-4fb9-8c5d-78225b269dcd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5315.400000000001"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "5906 * .9"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8eb0c2c6-2b7c-401b-b18d-373340748410",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
