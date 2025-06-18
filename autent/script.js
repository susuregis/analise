function login() {
  const email = document.getElementById('email').value;
  const senha = document.getElementById('senha').value;

  if (email === 'admin@teste.com' && senha === '123456') {
    const token = 'secrettoken123';
    window.location.href = 'http://localhost:8050?token=' + token;
  } else {
    document.getElementById('erro').textContent = 'Credenciais inválidas.';
  }
}
