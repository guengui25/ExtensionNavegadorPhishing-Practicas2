document.getElementById('analyzeBtn').addEventListener('click', () => {
  const text = document.getElementById('textInput').value.trim();
  const resultDiv = document.getElementById('result');

  // Clear previous result and ensure visibility
  resultDiv.style.display = "block"; 
  resultDiv.className = "";
  resultDiv.innerHTML = "";

  if (!text) {
    resultDiv.innerHTML = "<p style='color: red;'>Please enter some text.</p>";
    return;
  }
  
  // Display loading message
  resultDiv.innerHTML = "⏳ Processing...";
  resultDiv.classList.add("loading");

  // API call (make sure the URL matches the server's URL)
  fetch('http://localhost:9000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ 
      text: text,
      analyze_links: true
    })
  })
  .then(response => {
    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }
    return response.json();
  })
  .then(data => {
    // Clear previous result
    resultDiv.innerHTML = '';
    resultDiv.className = '';
    
    // Main section with analysis result
    const mainResultDiv = document.createElement('div');
    mainResultDiv.className = data.label === 'phishing' ? 'phishing' : 'safe';
    
    // Icon and main message
    if (data.label === 'phishing') {
      mainResultDiv.innerHTML = `
        <p>🚨 <strong>Warning:</strong> This text is likely <strong>phishing</strong> 
        (score: ${data.prediction.toFixed(2)})</p>
      `;
    } else {
      mainResultDiv.innerHTML = `
        <p>✅ <strong>Safe:</strong> This text is likely <strong>legitimate</strong> 
        (score: ${data.prediction.toFixed(2)})</p>
      `;
    }
    
    // Add language information as a small tag at the end of the main result
    if (data.language && data.language.code !== 'en' && data.language.translated) {
      const translatedBadge = document.createElement('span');
      translatedBadge.className = 'translated-badge';
      translatedBadge.innerHTML = `🌐 Translated from ${data.language.detected}`;
      translatedBadge.style.fontSize = '0.8em';
      translatedBadge.style.color = '#666';
      translatedBadge.style.display = 'block';
      translatedBadge.style.marginTop = '5px';
      mainResultDiv.appendChild(translatedBadge);
    }
    
    resultDiv.appendChild(mainResultDiv);
    
    // Create a map of domain safety for quick lookup
    const domainSafetyMap = {};
    if (data.domains && data.domain_analysis) {
      data.domains.forEach(domain => {
        if (data.domain_analysis[domain]) {
          domainSafetyMap[domain] = data.domain_analysis[domain].safe;
        }
      });
    }
    
    // Section for detected URLs with new analysis format
    if (data.urls && data.urls.length > 0) {
      const urlSection = document.createElement('div');
      urlSection.className = 'detail-section';
      urlSection.innerHTML = `<h3>🔗 URLs Detected (${data.urls.length})</h3>`;
      
      const urlList = document.createElement('ul');
      data.urls.forEach(url => {
        const urlItem = document.createElement('li');
        const analysis = data.url_analysis && data.url_analysis[url];
        
        let statusIcon = '⏳';
        let statusClass = 'pending';
        
        if (analysis) {
          if (analysis.error) {
            statusIcon = '❓';
            statusClass = 'error';
          } else if (analysis.status === 'pending_analysis') {
            statusIcon = '⏳';
            statusClass = 'pending';
          } else if (!analysis.safe) {
            statusIcon = '❌';
            statusClass = 'dangerous';
          } else {
            statusIcon = '✅';
            statusClass = 'safe';
          }
        }
        
        urlItem.innerHTML = `
          <span class="status ${statusClass}">${statusIcon}</span>
          <a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>
        `;
        urlList.appendChild(urlItem);
      });
      
      urlSection.appendChild(urlList);
      resultDiv.appendChild(urlSection);
    }
    
    // Section for detected domains with analysis
    if (data.domains && data.domains.length > 0) {
      const domainSection = document.createElement('div');
      domainSection.className = 'detail-section';
      domainSection.innerHTML = `<h3>🌐 Domains Detected (${data.domains.length})</h3>`;
      
      const domainList = document.createElement('ul');
      data.domains.forEach(domain => {
        const domainItem = document.createElement('li');
        const analysis = data.domain_analysis && data.domain_analysis[domain];
        
        let statusIcon = '⏳';
        let statusClass = 'pending';
        
        if (analysis) {
          if (analysis.error) {
            statusIcon = '❓';
            statusClass = 'error';
          } else if (analysis.status === 'pending_analysis') {
            statusIcon = '⏳';
            statusClass = 'pending';
          } else if (!analysis.safe) {
            statusIcon = '❌';
            statusClass = 'dangerous';
          } else {
            statusIcon = '✅';
            statusClass = 'safe';
          }
        }
        
        domainItem.innerHTML = `
          <span class="status ${statusClass}">${statusIcon}</span>
          ${domain}
        `;
        domainList.appendChild(domainItem);
      });
      
      domainSection.appendChild(domainList);
      resultDiv.appendChild(domainSection);
    }
    
    // Section for detected emails - updated to check domain safety
    if (data.emails && data.emails.length > 0) {
      const emailSection = document.createElement('div');
      emailSection.className = 'detail-section';
      emailSection.innerHTML = `<h3>📧 Email Addresses Detected (${data.emails.length})</h3>`;
      
      const emailList = document.createElement('ul');
      data.emails.forEach(email => {
        const emailItem = document.createElement('li');
        
        // Extract domain from email for safety check
        const emailDomain = email.split('@')[1];
        let statusIcon = '📧';
        let statusClass = 'neutral';
        
        // Check if domain is marked as unsafe in our domain safety map
        if (emailDomain && domainSafetyMap.hasOwnProperty(emailDomain)) {
          if (domainSafetyMap[emailDomain] === false) {
            // Domain is marked as unsafe
            statusIcon = '❌';
            statusClass = 'dangerous';
          } else {
            // Domain is marked as safe
            statusIcon = '✅';
            statusClass = 'safe';
          }
        }
        
        emailItem.innerHTML = `
          <span class="status ${statusClass}">${statusIcon}</span>
          ${email} ${statusClass === 'dangerous' ? '<span style="color:red">(Unsafe Domain)</span>' : ''}
        `;
        emailList.appendChild(emailItem);
      });
      
      emailSection.appendChild(emailList);
      resultDiv.appendChild(emailSection);
    }
    
    // If there are no URLs, domains or emails
    if ((!data.urls || data.urls.length === 0) && 
        (!data.domains || data.domains.length === 0) && 
        (!data.emails || data.emails.length === 0)) {
      const noLinksDiv = document.createElement('div');
      noLinksDiv.className = 'detail-section';
      noLinksDiv.innerHTML = '<p>No URLs, domains, or emails were detected in this text.</p>';
      resultDiv.appendChild(noLinksDiv);
    }
  })
  .catch(err => {
    console.error("API Connection Error:", err);
    resultDiv.classList.add("phishing");
    resultDiv.innerHTML = "<p style='color: red;'>❌ Connection error: Unable to reach the API.</p>";
  });
});