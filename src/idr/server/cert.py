"""Utility to generate a self-signed TLS/SSL certificate for local mobile development."""

import datetime
import ipaddress
import os
from pathlib import Path
import socket
from typing import List, Tuple
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def get_local_ip_addresses() -> List[str]:
    """Retrieve all local IPv4 addresses."""
    ips = ["127.0.0.1", "localhost"]
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    return ips


def generate_self_signed_cert(
    cert_dir: str = "certs",
    cert_filename: str = "server_cert.pem",
    key_filename: str = "server_key.pem",
) -> Tuple[Path, Path]:
    """Generate or retrieve a self-signed SSL certificate with SANs."""
    target_dir = Path(cert_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    cert_path = target_dir / cert_filename
    key_path = target_dir / key_filename

    if cert_path.exists() and key_path.exists():
        return cert_path, key_path

    # Generate Private Key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )

    # Prepare SANs
    san_list: List[x509.GeneralName] = []
    for ip_str in get_local_ip_addresses():
        try:
            san_list.append(x509.IPAddress(ipaddress.ip_address(ip_str)))
        except ValueError:
            san_list.append(x509.DNSName(ip_str))

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "IN"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "IDR Navigation System"),
        x509.NameAttribute(NameOID.COMMON_NAME, "IDR Local Dev Server"),
    ])

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName(san_list),
            critical=False,
        )
        .sign(private_key, hashes.SHA256(), default_backend())
    )

    # Write Key
    with open(key_path, "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    # Write Cert
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    return cert_path, key_path


if __name__ == "__main__":
    cert_p, key_p = generate_self_signed_cert()
    print(f"Self-signed SSL Certificate generated:")
    print(f"  Cert: {cert_p.resolve()}")
    print(f"  Key:  {key_p.resolve()}")
