document.addEventListener("DOMContentLoaded", () => {
  // Organize milestones into snake-like rows with animations
  const organizeRoadmap = () => {
    const container = document.querySelector(".milestone-container");
    if (!container) return;

    const milestones = Array.from(container.querySelectorAll(".milestone"));
    if (milestones.length === 0) return;

    // Clear container
    container.innerHTML = "";

    // Maximum milestones per row
    const milestonesPerRow = Math.min(4, Math.ceil(milestones.length / 2));

    // Calculate how many rows we need
    const rowCount = Math.ceil(milestones.length / milestonesPerRow);

    // Create rows and distribute milestones with animations
    for (let i = 0; i < rowCount; i++) {
      const row = document.createElement("div");
      row.className = "milestone-row";

      // Delay row appearance for staggered animation
      row.style.opacity = "0";
      row.style.transition = "opacity 0.5s ease-out";

      // Set a delay based on row index
      setTimeout(() => {
        row.style.opacity = "1";
      }, i * 600);

      // Add road elements to each row
      const roadPath = document.createElement("div");
      roadPath.className = "road-path";
      row.appendChild(roadPath);

      // Add road markings
      const roadMarking = document.createElement("div");
      roadMarking.className = "road-marking";
      roadPath.appendChild(roadMarking);

      // Add curves between rows if not the last row
      if (i < rowCount - 1) {
        const curveClass = i % 2 === 0 ? "curve-right" : "curve-left";

        // Current row curve
        const curve = document.createElement("div");
        curve.className = `road-curve ${curveClass}`;
        curve.style.zIndex = "1"; // Ensure curve is behind milestones
        row.appendChild(curve);

        // Prepare the vertical connector for the next row
        const connector = document.createElement("div");
        connector.className = "road-vertical-connector";
        connector.style.position = "absolute";
        connector.style.width = "35px";
        connector.style.height = "60px";
        connector.style.backgroundColor = "#1c1c1c";
        connector.style.bottom = "-60px";
        connector.style.zIndex = "1";

        if (i % 2 === 0) {
          connector.style.right = "0";
        } else {
          connector.style.left = "0";
        }

        row.appendChild(connector);
      }

      const startIndex = i * milestonesPerRow;
      const endIndex = Math.min((i + 1) * milestonesPerRow, milestones.length);

      // If odd row (zero-indexed), reverse the milestones
      const rowMilestones = milestones.slice(startIndex, endIndex);
      if (i % 2 === 1) {
        rowMilestones.reverse();
      }

      // Add animation delay for each milestone
      rowMilestones.forEach((milestone, j) => {
        milestone.style.position = "relative";

        // Set animation delay based on position
        milestone.style.animationDelay = `${i * 600 + j * 200}ms`;

        // Also set delay for the milestone number
        const milestoneNumber = milestone.querySelector(".milestone-number");
        if (milestoneNumber) {
          milestoneNumber.style.animationDelay = `${i * 600 + j * 200 + 300}ms`;
        }

        row.appendChild(milestone);
      });

      container.appendChild(row);
    }
  };

  // Run the organizer
  organizeRoadmap();

  // Add "Take Test" buttons to milestones
  addTestButtonsToMilestones();

  // Create test modal container if it doesn't exist
  createTestModal();

  // Add smooth scrolling for navigation
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault();
      const targetId = this.getAttribute("href");
      const targetElement = document.querySelector(targetId);

      if (targetElement) {
        window.scrollTo({
          top: targetElement.offsetTop - 100,
          behavior: "smooth",
        });
      }
    });
  });

  // Form validation and loading state
  const searchForm = document.querySelector("form");
  if (searchForm) {
    searchForm.addEventListener("submit", (e) => {
      const input = document.querySelector("#role");
      if (!input.value.trim()) {
        e.preventDefault();
        input.classList.add("error");

        // Add error message if it doesn't exist
        if (!document.querySelector(".error-message")) {
          const errorMsg = document.createElement("div");
          errorMsg.className = "error-message";
          errorMsg.textContent = "Please enter a career role";
          errorMsg.style.color = "#ef4444";
          errorMsg.style.marginTop = "5px";
          searchForm.appendChild(errorMsg);
        }
      } else {
        // Show loading animation
        const loadingElement = document.getElementById("loading");
        if (loadingElement) {
          loadingElement.style.display = "block";
        }

        // Hide the form during loading
        searchForm.style.opacity = "0.5";
        searchForm.style.pointerEvents = "none";

        // Disable the button
        const submitButton = document.getElementById("generateBtn");
        if (submitButton) {
          submitButton.disabled = true;
          submitButton.innerHTML =
            '<i class="fas fa-spinner fa-spin"></i> Generating...';
        }
      }
    });

    // Remove error class on input
    const input = document.querySelector("#role");
    if (input) {
      input.addEventListener("input", () => {
        input.classList.remove("error");
        const errorMsg = document.querySelector(".error-message");
        if (errorMsg) {
          errorMsg.remove();
        }
      });
    }
  }

  // Wait for the document to be ready
  document.addEventListener("DOMContentLoaded", function () {
    // Set up event listener for the generate button
    const generateBtn = document.getElementById("generateBtn");
    if (generateBtn) {
      generateBtn.addEventListener("click", function (e) {
        // Only show loading if the form is valid
        const roleInput = document.getElementById("role");
        if (roleInput && roleInput.value.trim()) {
          showLoading();
        }
      });
    }

    // Load completed milestones from localStorage
    loadCompletedMilestones();
  });
});

// Function to add "Take Test" buttons to each milestone
function addTestButtonsToMilestones() {
  const milestones = document.querySelectorAll(".milestone");

  milestones.forEach((milestone) => {
    // Check if milestone already has a test button
    if (!milestone.querySelector(".take-test-btn")) {
      const testBtn = document.createElement("button");
      testBtn.className = "take-test-btn";
      testBtn.innerHTML = '<i class="fas fa-vial"></i> Take Test';
      testBtn.style.display = "none"; // Hide by default

      testBtn.addEventListener("click", function (e) {
        e.stopPropagation(); // Prevent the milestone click event

        // Get the stage name from the milestone
        const stageName = milestone
          .querySelector(".milestone-title")
          .textContent.trim();

        // Show test modal and generate test
        showTestModal(stageName);
      });

      milestone.appendChild(testBtn);

      // Add hover event listeners
      milestone.addEventListener("mouseenter", function () {
        testBtn.style.display = "flex";
      });

      milestone.addEventListener("mouseleave", function () {
        testBtn.style.display = "none";
      });
    }
  });
}

// Function to load completed milestones from localStorage
function loadCompletedMilestones() {
  try {
    const completedMilestones = JSON.parse(
      localStorage.getItem("completedMilestones") || "[]"
    );

    // Mark each completed milestone
    completedMilestones.forEach((stageName) => {
      markMilestoneCompleted(stageName);
    });
  } catch (e) {
    console.error("Error loading completed milestones:", e);
  }
}

// Function to create the test modal
function createTestModal() {
  // Check if the modal already exists
  if (!document.getElementById("testModal")) {
    const modal = document.createElement("div");
    modal.id = "testModal";
    modal.className = "test-modal";

    // Create initial content structure
    modal.innerHTML = `
            <div class="test-content">
                <div class="test-header">
                    <h3>Skill Assessment Test</h3>
                    <button class="close-test-btn">&times;</button>
                </div>
                <div class="test-body" id="testBody">
                    <div class="test-loading">
                        <div class="test-spinner"></div>
                        <p>Preparing your assessment...</p>
                    </div>
                </div>
                <div class="test-footer">
                    <button class="test-submit-btn" id="submitTestBtn" style="display: none;">Submit Answers</button>
                </div>
            </div>
        `;

    document.body.appendChild(modal);

    // Add event listener to close button
    modal.querySelector(".close-test-btn").addEventListener("click", () => {
      closeTestModal();
    });

    // Close modal when clicking outside
    modal.addEventListener("click", function (e) {
      if (e.target === modal) {
        closeTestModal();
      }
    });
  }
}

// Function to show the test modal and generate test questions
function showTestModal(stageName) {
  const modal = document.getElementById("testModal");
  const testBody = document.getElementById("testBody");
  const submitBtn = document.getElementById("submitTestBtn");

  // Update modal title
  modal.querySelector(
    ".test-header h3"
  ).textContent = `${stageName} - Skill Assessment`;

  // Show loading state
  testBody.innerHTML = `
        <div class="test-loading">
            <div class="test-spinner"></div>
            <p>Generating questions for ${stageName}...</p>
        </div>
    `;
  submitBtn.style.display = "none";

  // Show the modal
  modal.classList.add("active");

  // Generate test questions
  generateTestQuestions(stageName)
    .then((questions) => {
      displayTestQuestions(questions, stageName);
      submitBtn.style.display = "block";

      // Add event listener to submit button
      submitBtn.onclick = function () {
        evaluateTest(questions, stageName);
      };
    })
    .catch((error) => {
      testBody.innerHTML = `
                <div class="test-error">
                    <p>Error generating questions. Please try again.</p>
                    <p class="error-details">${error.message}</p>
                </div>
            `;
    });
}

// Function to close the test modal
function closeTestModal() {
  const modal = document.getElementById("testModal");
  if (modal) {
    modal.classList.remove("active");
  }
}

// Function to generate test questions using Gemini API directly
// Function to generate test questions using backend endpoint
async function generateTestQuestions(stageName) {
  try {
    console.log("Generating test questions for:", stageName);

    // Use backend endpoint instead of calling API directly
    const response = await fetch("/generate-test", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        topic: stageName,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || "Failed to generate questions");
    }

    const data = await response.json();
    return data.questions;
  } catch (error) {
    console.error("Error generating questions:", error);
    throw error;
  }
}

// Function to display the test questions
function displayTestQuestions(questions, stageName) {
  const testBody = document.getElementById("testBody");

  // Create HTML for questions
  let questionsHTML = `
        <div class="test-instructions">
            <p>Answer the following questions about <strong>${stageName}</strong> to assess your knowledge level. Your results will help determine if you should study this topic or can confidently move to the next stage.</p>
        </div>
    `;

  questions.forEach((question, index) => {
    // Clean up question text - remove question number prefix if it exists
    let questionText = question.question;
    questionText = questionText
      .replace(/^(\d+\.\s*|\*\*\d+\.\*\*\s*|Question \d+:)/i, "")
      .trim();

    questionsHTML += `
            <div class="test-question" data-question-index="${index}">
                <div class="question-text">${index + 1}. ${questionText}</div>
                <div class="options-container">
        `;

    // Loop through options and create radio inputs
    question.options.forEach((option, optIndex) => {
      // Clean up option text - extract just the content
      let optionText = option;
      const optionLetter = String.fromCharCode(65 + optIndex);

      // Remove option prefix if it exists (A), B), etc.)
      optionText = optionText.replace(/^[A-D][\)\.]?\s*/i, "").trim();

      const optionId = `q${index}_opt${optIndex}`;
      questionsHTML += `
                <label class="option-label" for="${optionId}">
                    <input type="radio" id="${optionId}" name="q${index}" value="${optionLetter}">
                    <span>${optionLetter}) ${optionText}</span>
                </label>
            `;
    });

    questionsHTML += `
                </div>
            </div>
        `;
  });

  testBody.innerHTML = questionsHTML;

  // Add event listeners for radio inputs to provide visual feedback
  document
    .querySelectorAll('.option-label input[type="radio"]')
    .forEach((radio) => {
      radio.addEventListener("change", function () {
        // Remove unanswered class when an option is selected
        const questionElement = this.closest(".test-question");
        if (questionElement) {
          questionElement.classList.remove("unanswered");
        }
      });
    });
}

// Function to evaluate the test answers locally
function evaluateTestLocally(questions, answers) {
  let correctCount = 0;
  const details = [];

  // Process each question
  for (let i = 0; i < questions.length; i++) {
    const question = questions[i];
    const userAnswer = answers[i];

    // User selection is A,B,C,D but we need to map it to the actual option content
    let userSelectedOption = null;
    if (userAnswer !== null) {
      const optionIndex = userAnswer.charCodeAt(0) - 65; // 'A' -> 0, 'B' -> 1, etc.
      if (optionIndex >= 0 && optionIndex < question.options.length) {
        userSelectedOption = question.options[optionIndex];
      }
    }

    // Compare with correct answer - check both possible property names
    const correctAnswer = question.correctAnswer || question.correct_answer;
    const isCorrect = userAnswer === correctAnswer;
    if (isCorrect) {
      correctCount++;
    }

    // Add to details
    details.push({
      question_number: i + 1,
      is_correct: isCorrect,
      user_answer: userSelectedOption || "Not answered",
      correct_answer: correctAnswer,
    });
  }

  // Calculate score
  const score = ((correctCount / questions.length) * 100).toFixed(1);
  const isPassing = correctCount >= Math.ceil(questions.length * 0.7); // 70% passing threshold

  // Generate recommendation
  let recommendation = "";
  if (isPassing) {
    recommendation = `Congratulations! You've achieved a score of ${score}%. You have a good understanding of the core concepts in this topic and can proceed to the next stage.`;
  } else {
    recommendation = `You scored ${score}%. We recommend spending more time studying this topic before moving forward. Focus on understanding the key concepts better.`;
  }

  return {
    score: score + "%",
    correct_count: correctCount,
    total_questions: questions.length,
    answered_count: answers.filter((a) => a !== null).length,
    passed: isPassing,
    recommendation: recommendation,
    details: details,
  };
}

// Updated evaluation function to use local evaluation when API is not available
async function evaluateTest(questions, stageName) {
  // Collect user answers
  const answers = [];

  // Check if all questions are answered
  let allAnswered = true;

  questions.forEach((_, index) => {
    const selectedOption = document.querySelector(
      `input[name="q${index}"]:checked`
    );

    if (selectedOption) {
      answers.push(selectedOption.value);
    } else {
      answers.push(null);
      allAnswered = false;

      // Highlight unanswered question
      document
        .querySelector(`[data-question-index="${index}"]`)
        .classList.add("unanswered");
    }
  });

  if (!allAnswered) {
    alert("Please answer all questions before submitting.");
    return;
  }

  // Show loading state
  const testBody = document.getElementById("testBody");
  const submitBtn = document.getElementById("submitTestBtn");

  testBody.innerHTML = `
        <div class="test-loading">
            <div class="test-spinner"></div>
            <p>Evaluating your answers...</p>
        </div>
    `;
  submitBtn.style.display = "none";

  try {
    // Evaluate locally without needing the API
    const result = evaluateTestLocally(questions, answers);
    displayTestResults(result, stageName);

    // If user passed, mark this milestone as completed
    if (result.passed) {
      markMilestoneCompleted(stageName);
    }
  } catch (error) {
    console.error("Error evaluating test:", error);
    testBody.innerHTML = `
            <div class="test-error">
                <p>Error evaluating your answers. Please try again.</p>
                <p class="error-details">${error.message}</p>
            </div>
        `;
  }
}

// Function to display test results
function displayTestResults(result, stageName) {
  const testBody = document.getElementById("testBody");
  const score = parseFloat(result.score.replace("%", ""));
  const isPassing = score >= 70;

  // Get correct count and total questions from result if available
  const correctCount = result.correct_count || 0;
  const totalQuestions = result.total_questions || 20;
  const answeredCount = result.answered_count || totalQuestions;

  testBody.innerHTML = `
        <div class="test-result">
            <h3>Assessment Results for ${stageName}</h3>
            <div class="result-score">${result.score}</div>
            <p>You answered ${correctCount} out of ${answeredCount} questions correctly</p>
            <div class="result-message ${isPassing ? "pass" : "fail"}">
                ${result.recommendation}
            </div>
            <div class="result-action">
                ${
                  isPassing
                    ? `<p>Great job! You have demonstrated proficiency in this topic. 
                    You can confidently move to the next stage in your learning journey.</p>`
                    : `<p>We recommend spending more time studying this topic before moving forward.
                    Focus on understanding the core concepts better.</p>`
                }
                <button class="test-close-btn">Close Assessment</button>
            </div>
        </div>
    `;

  // Add event listener to close button
  testBody.querySelector(".test-close-btn").addEventListener("click", () => {
    closeTestModal();
  });
}

// Function to mark a milestone as completed
function markMilestoneCompleted(stageName) {
  const milestones = document.querySelectorAll(".milestone");

  milestones.forEach((milestone) => {
    const title = milestone.querySelector(".milestone-title");
    if (title && title.textContent.trim() === stageName) {
      // Add completed class
      milestone.classList.add("completed-milestone");

      // Add checkmark if not already there
      if (!milestone.querySelector(".milestone-completed")) {
        const checkmark = document.createElement("div");
        checkmark.className = "milestone-completed";
        checkmark.innerHTML = '<i class="fas fa-check-circle"></i>';
        milestone.appendChild(checkmark);
      }

      // Save to localStorage
      saveCompletedMilestone(stageName);
    }
  });
}

// Function to save completed milestone to localStorage
function saveCompletedMilestone(stageName) {
  try {
    let completedMilestones = JSON.parse(
      localStorage.getItem("completedMilestones") || "[]"
    );
    if (!completedMilestones.includes(stageName)) {
      completedMilestones.push(stageName);
      localStorage.setItem(
        "completedMilestones",
        JSON.stringify(completedMilestones)
      );
    }
  } catch (e) {
    console.error("Error saving completed milestone:", e);
  }
}
