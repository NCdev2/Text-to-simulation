# Procedure: Enabling and Generating Service Account JSON Keys for a Specific GCP Project (When Org Policy iam.disableServiceAccountKeyCreation is Active)

**Objective:** To allow and subsequently generate a service account JSON key file for a designated Google Cloud Project (e.g., `YOUR_PROJECT_NAME` or `YOUR_PROJECT_ID`) when the organization-level policy `iam.disableServiceAccountKeyCreation` prevents it by default.

**Prerequisites:**

*   The user performing Part 1 must have Organization Policy Administrator privileges (`roles/orgpolicy.policyAdmin`) at the appropriate level (Organization or a Folder containing the target project).
*   The user performing Part 2 must have permissions to manage service accounts and create keys within the target project (e.g., "Service Account Key Admin" or "Owner" on the service account).

## Part 1: Modifying Organization Policy to Allow Key Creation for the Specific Project

*(To be performed by an Organization Policy Administrator)*

1.  **Navigate to Organization Policies:**
    *   In the Google Cloud Console, go to "IAM & Admin" > "Organization Policies."
    *   Ensure you are viewing policies at the Organization or relevant Folder level.
2.  **Locate and Select Policy:**
    *   Find the policy named "Disable Service Account Key Creation" (Constraint ID: `constraints/iam.disableServiceAccountKeyCreation`).
    *   Click on it to manage the policy.
3.  **Edit Policy for Target Project:**
    *   Click "Manage policy" or "Edit policy."
    *   In the editing interface, specify that the modification "Applies to" your target project: `YOUR_PROJECT_NAME` (or `YOUR_PROJECT_ID`).
4.  **Override Parent's Policy:**
    *   Select the option similar to "Override parent's policy" for the policy source as it applies to `YOUR_PROJECT_NAME`.
5.  **Set Enforcement to "Not Enforced":**
    *   For the overridden policy on `YOUR_PROJECT_NAME`, configure the "Status" or "Policy enforcement" for the rule to be "Not enforced". This action means the restriction (disabling key creation) will not apply to this specific project.
6.  **Save Policy Changes:**
    *   Click "Set policy" or "Save" to apply the changes. Allow a few minutes for propagation.

## Part 2: Generating the Service Account JSON Key within the Enabled Project

*(To be performed by a user with necessary permissions within `YOUR_PROJECT_NAME`)*

1.  **Navigate to Service Accounts:**
    *   In the Google Cloud Console, go to "IAM & Admin" > "Service Accounts."
2.  **Select Target Project:**
    *   Ensure the project selected in the console is `YOUR_PROJECT_NAME` (or `YOUR_PROJECT_ID`).
3.  **Choose or Create Service Account:**
    *   Select the specific service account for which you need a key (e.g., `your-service-account-email@YOUR_PROJECT_ID.iam.gserviceaccount.com`).
    *   If one doesn't exist, create a new service account, granting it the appropriate IAM roles for its intended purpose (e.g., "Vertex AI User" for accessing Vertex AI).
4.  **Access Keys Tab:**
    *   Once the service account is selected/created, navigate to its "KEYS" tab.
5.  **Add New Key:**
    *   Click "ADD KEY".
    *   Select "Create new key" from the dropdown.
6.  **Choose Key Type and Create:**
    *   Select "JSON" as the key type.
    *   Click "CREATE".
    *   The service account JSON key file (e.g., `GENERATED_KEY_FILENAME.json`) will be automatically downloaded to your local machine.

## Part 3: Secure Handling of the Generated JSON Key

*   **Confidentiality:** Treat the downloaded `GENERATED_KEY_FILENAME.json` file as highly confidential. It contains credentials that grant access to your Google Cloud resources as the service account.
*   **No Hardcoding:** Do NOT embed the key directly in source code or commit it to version control systems (like Git).
*   **Secure Storage and Usage:**
    *   For local development, set the path to the key file in the `GOOGLE_APPLICATION_CREDENTIALS` environment variable.
    *   For deployed applications (e.g., Streamlit), use built-in secrets management tools provided by the hosting platform (e.g., Streamlit secrets, Google Secret Manager, or environment variables injected by the cloud provider).
    *   Store the actual file securely or its contents within a secret manager.
