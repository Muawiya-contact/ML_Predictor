async function getPrediction() {
  const data = {
    "SBP": document.getElementById("SBP").value,
    "DBP": document.getElementById("DBP").value,
    "HR": document.getElementById("HR").value,
    "RR": document.getElementById("RR").value,
    "BT": document.getElementById("BT").value,
    "SpO2": document.getElementById("SpO2").value,
    "Age": document.getElementById("Age").value,
    "Gender": document.getElementById("Gender").value,
    "GCS": document.getElementById("GCS").value,
    "Na": document.getElementById("Na").value,
    "K": document.getElementById("K").value,
    "Cl": document.getElementById("Cl").value,
    "Urea": document.getElementById("Urea").value,
    "Creatinine": document.getElementById("Creatinine").value,
    "Alcoholic": document.getElementById("Alcoholic").value,
    "Smoke": document.getElementById("Smoke").value,
    "FHCD": document.getElementById("FHCD").value,
    "TriageScore": document.getElementById("TriageScore").value
  };

  const response = await fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"   // ✅ must be json
    },
    body: JSON.stringify(data)             // ✅ stringify payload
  });

  const result = await response.json();
  document.getElementById("output").innerHTML =
    `<div class="result-box">
       <h2>Prediction Result</h2>
       <p><b>Prediction:</b> ${result.prediction}</p>
       <p><b>Explanation:</b> ${result.explanation}</p>
     </div>`;
}