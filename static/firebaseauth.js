import { initializeApp } from "https://www.gstatic.com/firebasejs/10.11.1/firebase-app.js";
import { getAuth, createUserWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/10.11.1/firebase-auth.js";
import { getFirestore, doc, setDoc } from "https://www.gstatic.com/firebasejs/10.11.1/firebase-firestore.js";

// Firebase Config
const firebaseConfig = {
  apiKey: "AIzaSyB3nU-WVQ-7zm9RijUhj420QPET9GYQpj8",
  authDomain: "login-form-84016.firebaseapp.com",
  projectId: "login-form-84016",
  storageBucket: "login-form-84016.firebasestorage.app",
  messagingSenderId: "568483509830",
  appId: "1:568483509830:web:0d6de2a1537154bd67fc47",
  measurementId: "G-KCGTMGB6HB"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);

let currentStep = 1;
const totalSteps = 4;

const progressFill = document.getElementById("progressFill");
const prevBtn = document.getElementById("prevBtn");
const nextBtn = document.getElementById("nextBtn");
const signUpBtn = document.getElementById("signUpBtn");
const roleSelect = document.getElementById("role");
const referralGroup = document.getElementById("referralGroup");

function showStep(step) {
  document.querySelectorAll(".form-step").forEach((el, idx) => {
    el.classList.toggle("active", idx === step - 1);
  });
  document.querySelectorAll(".step-circle").forEach((el, idx) => {
    el.classList.toggle("active", idx < step);
  });
  document.querySelectorAll(".step-label").forEach((el, idx) => {
    el.classList.toggle("active", idx < step);
  });
  progressFill.style.width = ((step - 1) / (totalSteps - 1)) * 100 + "%";
  prevBtn.style.display = step > 1 ? "inline-flex" : "none";
  nextBtn.style.display = step < totalSteps ? "inline-flex" : "none";
  signUpBtn.style.display = step === totalSteps ? "inline-flex" : "none";
}

function validateStep(step) {
  let valid = true;
  const stepDiv = document.getElementById("step" + step);
  stepDiv.querySelectorAll("input, select").forEach(input => {
    if (!input.value) {
      input.classList.add("error");
      valid = false;
    } else {
      input.classList.remove("error");
    }
  });
  if (step === 2) {
    const pw = document.getElementById("rPassword").value;
    const cpw = document.getElementById("confirmPassword").value;
    if (pw !== cpw || pw.length < 6) {
      document.getElementById("rPassword").classList.add("error");
      document.getElementById("confirmPassword").classList.add("error");
      valid = false;
    }
  }
  return valid;
}

prevBtn.addEventListener("click", () => {
  if (currentStep > 1) {
    currentStep--;
    showStep(currentStep);
  }
});

nextBtn.addEventListener("click", () => {
  if (validateStep(currentStep)) {
    if (currentStep < totalSteps) {
      currentStep++;
      showStep(currentStep);
    }
  }
});

roleSelect.addEventListener("change", () => {
  referralGroup.classList.toggle("hidden", roleSelect.value === "staff" || roleSelect.value === "admin" ? false : true);
});

function showMessage(message, isError = false) {
  const msgDiv = document.getElementById("signUpMessage");
  msgDiv.textContent = message;
  msgDiv.style.backgroundColor = isError ? "#f8d7da" : "#d4edda";
  msgDiv.style.color = isError ? "#721c24" : "#155724";
  msgDiv.style.display = "block";
  msgDiv.style.opacity = 1;
  setTimeout(() => { msgDiv.style.opacity = 0; }, 5000);
}

// ✅ Fixed handleSignUp
window.handleSignUp = async function () {
  let valid = true;
  for (let step = 1; step <= 4; step++) {
    if (!validateStep(step)) {
      valid = false;
      currentStep = step;
      showStep(currentStep);
      break;
    }
  }
  if (!valid) {
    showMessage("⚠️ Please complete all steps correctly.", true);
    return;
  }

  const referralInput = document.getElementById("referralCode");
  if (roleSelect.value === "admin" || roleSelect.value === "staff") {
    const requiredCode = roleSelect.value === "admin" ? "541321" : "123145";
    if (referralInput.value.trim() !== requiredCode) {
      showMessage(`❌ Invalid referral code for ${roleSelect.value}!`, true);
      referralInput.classList.add("error");
      currentStep = 3;
      showStep(currentStep);
      return;
    }
  }

  document.getElementById("loadingOverlay").style.display = "flex";

  const firstName = document.getElementById("fName").value.trim();
  const lastName = document.getElementById("lName").value.trim();
  const phone = document.getElementById("phone").value.trim();
  const email = document.getElementById("rEmail").value.trim();
  const address = document.getElementById("address").value.trim();
  const password = document.getElementById("rPassword").value;
  const jobPosition = document.getElementById("jobPosition").value.trim();
  const department = document.getElementById("department").value.trim();
  const startDate = document.getElementById("startDate").value;
  const emergencyName = document.getElementById("emergencyName").value.trim();
  const emergencyPhone = document.getElementById("emergencyPhone").value.trim();
  const relation = document.getElementById("relation").value.trim();
  const role = roleSelect.value;

  try {
    const userCredential = await createUserWithEmailAndPassword(auth, email, password);
    await setDoc(doc(db, "users", userCredential.user.uid), {
      address,
      createdAt: new Date(),
      department,
      email,
      emergencyContact: { name: emergencyName, phone: emergencyPhone, relation },
      firstName,
      jobPosition,
      lastName,
      phone,
      role,
      startDate
    });

    showMessage("✅ Account created successfully! Redirecting...", false);
    // 🔥 redirect terus ke dashboard.html
    setTimeout(() => window.location.href = "/dashboard", 2500);

  } catch (error) {
    document.getElementById("loadingOverlay").style.display = "none";
    if (error.code === "auth/email-already-in-use") {
      showMessage("❌ Email already registered!", true);
    } else if (error.code === "auth/weak-password") {
      showMessage("❌ Password too weak!", true);
    } else {
      showMessage("❌ Registration failed: " + error.message, true);
    }
  }
};
