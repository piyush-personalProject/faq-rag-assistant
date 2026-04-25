# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue,
please report it responsibly.

### How to Report

**Please DO NOT file a public GitHub issue for security vulnerabilities.**

Instead, please report them via one of these methods:

1. **Email**: Send to the maintainers directly (if known contact exists)
2. **GitHub Security Advisories**: Use the [Security Advisories](https://github.com/YOUR_USERNAME/knowledgeStore/security/advisories) feature
3. **Private Vulnerability Reporting**: Use GitHub's [private vulnerability reporting](https://github.com/YOUR_USERNAME/knowledgeStore/security/advisories/new)

### What to Include

Please include as much of the following as possible:

- Type of vulnerability (e.g., SQL injection, XSS, etc.)
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

## Response Timeline

- **Initial Response**: Within 48 hours
- **Assessment**: Within 7 days
- **Fix Development**: Depends on severity
- **Disclosure**: After fix is available

## Security Best Practices for Users

If you deploy this project, follow these security best practices:

### Environment Variables

- Never commit `.env` files to version control
- Use strong, unique values for secrets
- Rotate credentials periodically

### Dependencies

```bash
# Audit dependencies for vulnerabilities
pip audit

# Update dependencies regularly
pip list --outdated
pip install -U package_name
```

### Deployment

- Run with minimal privileges
- Use HTTPS in production
- Keep systems updated
- Monitor logs for suspicious activity

## Security Updates

Security updates will be released as patch versions (e.g., 1.1.1) and announced
in the changelog with a "Security" label.

## Attribution

Thank you for helping keep this project and its users safe!
