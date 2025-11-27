    // Firebase imports
    import { initializeApp } from "https://www.gstatic.com/firebasejs/10.11.1/firebase-app.js";
    import { getAuth, onAuthStateChanged, signOut } from "https://www.gstatic.com/firebasejs/10.11.1/firebase-auth.js";
    import { getFirestore, doc, getDoc, setDoc } from "https://www.gstatic.com/firebasejs/10.11.1/firebase-firestore.js";

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

    // DOM Elements
    const navLinks = document.getElementById("navLinks");
    const sidebarAvatar = document.getElementById("sidebarAvatar");
    const sidebarUserName = document.getElementById("sidebarUserName");
    const sidebarUserEmail = document.getElementById("sidebarUserEmail");
    const toggleSidebar = document.getElementById("toggleSidebar");
    const sidebar = document.querySelector(".sidebar");
    const loadingOverlay = document.getElementById("loadingOverlay");

    // Profile elements
    const profileFName = document.getElementById("profileFName");
    const profileLName = document.getElementById("profileLName");
    const profileEmail = document.getElementById("profileEmail");
    const profilePhone = document.getElementById("profilePhone");
    const profileAddress = document.getElementById("profileAddress");
    const profileJob = document.getElementById("profileJob");
    const profileDept = document.getElementById("profileDept");
    const profileStart = document.getElementById("profileStart");
    const profileRole = document.getElementById("profileRole");
    const emergencyName = document.getElementById("emergencyName");
    const emergencyPhone = document.getElementById("emergencyPhone");
    const emergencyRelation = document.getElementById("emergencyRelation");

    // Toggle sidebar
    toggleSidebar.addEventListener("click", () => {
      sidebar.classList.toggle("collapsed");
      const icon = toggleSidebar.querySelector("i");
      if (sidebar.classList.contains("collapsed")) {
        icon.className = "fas fa-chevron-right";
      } else {
        icon.className = "fas fa-chevron-left";
      }
    });

    // Add navigation link function
    function addNavLink(text, icon, link, isActive = false) {
      const li = document.createElement("a");
      li.className = `nav-item ${isActive ? 'active' : ''}`;
      li.href = link;
      li.innerHTML = `<i class="fas ${icon}"></i><span>${text}</span>`;
      navLinks.appendChild(li);
    }

    // Add logout function
    function addLogout() {
      const li = document.createElement("a");
      li.className = "nav-item logout";
      li.href = "#";
      li.id = "logoutLink";
      li.innerHTML = '<i class="fas fa-sign-out-alt"></i><span>Logout</span>';
      li.addEventListener("click", (e) => {
        e.preventDefault();
        signOut(auth).then(() => {
          window.location.href = "index.html";
        }).catch((error) => {
          console.error("Logout error:", error);
        });
      });
      navLinks.appendChild(li);
    }

    // Load profile data
    async function loadProfileData(user) {
      // SHOW loading overlay at the start
      loadingOverlay.style.display = "flex";
      
      try {
        const userDoc = await getDoc(doc(db, "users", user.uid));
        if (userDoc.exists()) {
          const data = userDoc.data();
          
          // Update sidebar user info
          const displayName = data.firstName || user.email.split("@")[0];
          sidebarUserName.textContent = displayName;
          sidebarUserEmail.textContent = user.email;
          sidebarAvatar.textContent = displayName.charAt(0).toUpperCase();
          
          // Update profile fields
          profileFName.textContent = data.firstName || "N/A";
          profileLName.textContent = data.lastName || "N/A";
          profileEmail.textContent = user.email;
          profilePhone.textContent = data.phone || "N/A";
          profileAddress.textContent = data.address || "N/A";
          profileJob.textContent = data.jobPosition || "N/A";
          profileDept.textContent = data.department || "N/A";
          profileStart.textContent = data.startDate || "N/A";
          profileRole.textContent = data.role || "staff";
          
          // Emergency contact
          const emergency = data.emergencyContact || {};
          emergencyName.textContent = emergency.name || "N/A";
          emergencyPhone.textContent = emergency.phone || "N/A";
          emergencyRelation.textContent = emergency.relation || "N/A";
          
          // Build navigation based on role
          const role = data.role || "staff";
          navLinks.innerHTML = "";
          
          if (role === "admin") {
            addNavLink("Dashboard", "fa-home", "admin-dashboard.html");
            addNavLink("Staff Management", "fa-users", "manage-staff.html");
            addNavLink("Staff Attendance Log", "fa-list", "admin-attendancelog.html");
            addNavLink("Staff Leave Requests", "fa-calendar-alt", "admin-leavecalendar.html");
            addNavLink("Admin Profile", "fa-user", "profile.html", true);
          } else {
            addNavLink("Dashboard", "fa-home", "dashboard.html");
            addNavLink("Attendance Log", "fa-list", "attendancelog.html");
            addNavLink("Leave Requests", "fa-calendar-alt", "leavecalendar.html");
            addNavLink("Profile", "fa-user", "profile.html", true);
          }
          
          addLogout();
          
          // HIDE loading overlay after successful load
          loadingOverlay.style.display = "none";
        }
      } catch (error) {
        console.error("Error loading profile:", error);
        // Show error message in loading overlay
        loadingOverlay.innerHTML = '<p>Error loading profile. Please try again.</p>';
        // Keep it visible to show the error
      }
    }

    // Check authentication state
    onAuthStateChanged(auth, async (user) => {
      if (!user) {
        // Redirect to login if not authenticated
        window.location.href = "index.html";
      } else {
        // Load profile data
        await loadProfileData(user);
      }
    });

    // Edit/Save functionality
    const editBtn = document.getElementById("editBtn");
    const saveBtn = document.getElementById("saveBtn");
    const viewFields = document.getElementById("viewFields");
    const editFields = document.getElementById("editFields");

    editBtn.addEventListener("click", () => {
      // Populate edit fields with current values
      editFields.querySelector("#editFName").value = profileFName.textContent;
      editFields.querySelector("#editLName").value = profileLName.textContent;
      editFields.querySelector("#editPhone").value = profilePhone.textContent;
      editFields.querySelector("#editAddress").value = profileAddress.textContent;
      editFields.querySelector("#editJob").value = profileJob.textContent;
      editFields.querySelector("#editDept").value = profileDept.textContent;
      editFields.querySelector("#editStart").value = profileStart.textContent;
      editFields.querySelector("#editRole").value = profileRole.textContent;
      editFields.querySelector("#editEmergencyName").value = emergencyName.textContent;
      editFields.querySelector("#editEmergencyPhone").value = emergencyPhone.textContent;
      editFields.querySelector("#editEmergencyRelation").value = emergencyRelation.textContent;

      // Switch to edit mode
      viewFields.style.display = "none";
      editFields.style.display = "block";
      editBtn.style.display = "none";
      saveBtn.style.display = "inline-block";
    });

    saveBtn.addEventListener("click", async () => {
      // Get current user
      const user = auth.currentUser;
      if (!user) return;

      try {
        // Get updated values
        const updatedData = {
          firstName: editFields.querySelector("#editFName").value,
          lastName: editFields.querySelector("#editLName").value,
          phone: editFields.querySelector("#editPhone").value,
          address: editFields.querySelector("#editAddress").value,
          jobPosition: editFields.querySelector("#editJob").value,
          department: editFields.querySelector("#editDept").value,
          startDate: editFields.querySelector("#editStart").value,
          emergencyContact: {
            name: editFields.querySelector("#editEmergencyName").value,
            phone: editFields.querySelector("#editEmergencyPhone").value,
            relation: editFields.querySelector("#editEmergencyRelation").value
          }
        };

        // Update Firestore
        await setDoc(doc(db, "users", user.uid), updatedData, { merge: true });

        // Update view fields
        profileFName.textContent = updatedData.firstName;
        profileLName.textContent = updatedData.lastName;
        profilePhone.textContent = updatedData.phone;
        profileAddress.textContent = updatedData.address;
        profileJob.textContent = updatedData.jobPosition;
        profileDept.textContent = updatedData.department;
        profileStart.textContent = updatedData.startDate;
        emergencyName.textContent = updatedData.emergencyContact.name;
        emergencyPhone.textContent = updatedData.emergencyContact.phone;
        emergencyRelation.textContent = updatedData.emergencyContact.relation;

        // Switch back to view mode
        viewFields.style.display = "block";
        editFields.style.display = "none";
        editBtn.style.display = "inline-block";
        saveBtn.style.display = "none";

        alert("Profile updated successfully!");
      } catch (error) {
        console.error("Error updating profile:", error);
        alert("Failed to update profile. Please try again.");
      }
    });