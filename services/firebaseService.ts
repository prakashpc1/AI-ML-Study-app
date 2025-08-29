
// This file is kept for configuration consistency but is no longer used for Authentication or user-specific data.
// All user data is now handled locally via services/profile.ts (localStorage) and services/dbService.ts (IndexedDB).

// IMPORTANT: Your web app's Firebase configuration was here.
// It's no longer needed for the app to function but might be useful if you re-add cloud features.
const firebaseConfig = {
  apiKey: "AIzaSy..._PLACEHOLDER",
  authDomain: "YOUR_PROJECT_ID.firebaseapp.com",
  projectId: "YOUR_PROJECT_ID",
  storageBucket: "YOUR_PROJECT_ID.appspot.com",
  messagingSenderId: "YOUR_SENDER_ID",
  appId: "YOUR_APP_ID",
};

export const isFirebaseProperlyConfigured = 
    firebaseConfig.projectId !== "YOUR_PROJECT_ID" && 
    firebaseConfig.apiKey !== "AIzaSy..._PLACEHOLDER";

if (!isFirebaseProperlyConfigured) {
    console.warn("Firebase is using placeholder credentials. Any feature requiring Firebase will not work.");
}