"""
Generates self-signed SSL certificate for local HTTPS testing across Wi-Fi devices.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import ipaddress
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.x509.oid import NameOID

def generate_self_signed_cert(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    key_path = output_dir / "key.pem"
    cert_path = output_dir / "cert.pem"

    # Generate RSA private key
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Subject and Issuer
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "IN"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "BHAVIK"),
        x509.NameAttribute(NameOID.COMMON_NAME, "BHAVIK-Local-HTTPS"),
    ])

    # Subject Alternative Names (SAN) for localhost and local network IPs
    san = x509.SubjectAlternativeName([
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        x509.IPAddress(ipaddress.IPv4Address("10.119.136.129")),
    ])

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
        .add_extension(san, critical=False)
        .sign(key, hashes.SHA256())
    )

    # Write private key
    with open(key_path, "wb") as f:
        f.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    # Write certificate
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    print(f"Self-signed SSL certificate generated at:")
    print(f"  Cert: {cert_path}")
    print(f"  Key:  {key_path}")
    return cert_path, key_path

if __name__ == "__main__":
    certs_dir = Path(__file__).resolve().parent.parent / "certs"
    generate_self_signed_cert(certs_dir)
