# Materials

* [Enabling SAML 2.0 Federated Users to Access the AWS Management Console](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_providers_enable-console-saml.html)

# SAML Login Process

![](https://docs.aws.amazon.com/IAM/latest/UserGuide/images/saml-based-sso-to-console.diagram.png)


1. Login with ADFS. ex) the portal URL is: https://ADFSServiceName/adfs/ls/IdpInitiatedSignOn.aspx

2. The portal verifies the user's identity in your organization.

3. The portal generates a SAML authentication response that includes assertions that identify the user and include attributes about the user.

4. The client browser is redirected to the AWS single sign-on endpoint and posts the SAML assertion.

5. The endpoint requests temporary security credentials on behalf of the user and creates a console sign-in URL that uses those credentials.

6. AWS sends the sign-in URL back to the client as a redirect.

7. The client browser is redirected to the AWS Management Console.

