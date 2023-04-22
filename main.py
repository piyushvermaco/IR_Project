from website import create_app
import sys
sys.path.append('/path/to/flask/module')

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
