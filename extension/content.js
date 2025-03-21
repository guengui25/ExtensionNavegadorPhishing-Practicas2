// Asynchronous function to send text to the API and get the prediction
async function analyzeText(text) {
  try {
    const response = await fetch('http://localhost:9000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ text: text })
    });
    const data = await response.json();
    return data.label; // Expected to return 'phishing' or 'legit'
  } catch (error) {
    console.error('Error analyzing text:', error);
    return null;
  }
}

// Function to analyze and highlight paragraphs on the page
(function() {
  // Select all paragraphs
  const paragraphs = document.querySelectorAll('p');
  paragraphs.forEach(async (elem) => {
    const text = elem.innerText;
    const prediction = await analyzeText(text);
    if (prediction === 'phishing') {
      // Highlight the paragraph with a red background and border for better visibility
      elem.style.backgroundColor = 'rgba(255, 0, 0, 0.3)';
      elem.style.border = '1px solid red';
      elem.title = "Possible phishing detected in this text.";
    }
  });
})();