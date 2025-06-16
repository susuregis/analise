from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)
app.secret_key = 'user'  # Usado para manter as sessões seguras


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"  # Página para redirecionar quando o usuário não estiver autenticado


users = {'usuario': {'password_hash': generate_password_hash('senha123')}}  # Senha criptografada

# Classe de usuário para o Flask-Login
class User(UserMixin):
    def __init__(self, id):
        self.id = id


@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        return User(user_id)
    return None

# Página inicial (apenas acessível para usuários logados)
@app.route('/')
@login_required
def home():
    return f'Olá, {current_user.id}! Você está autenticado.'

# Página de login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Verificando as credenciais
        if username in users and check_password_hash(users[username]['password_hash'], password):
            user = User(username)
            login_user(user)  # Logando o usuário
            return redirect(url_for('home'))  # Redireciona para a página principal
        else:
            flash('Credenciais inválidas!', 'danger')  # Exibe mensagem de erro se as credenciais forem inválidas

    return render_template('login.html')

# Página de logout
@app.route('/logout')
@login_required
def logout():
    logout_user()  # Faz logout do usuário
    return redirect(url_for('login'))  # Redireciona para a página de login

if __name__ == '__main__':
    app.run(debug=True)
