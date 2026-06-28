from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return '<h1>Aircraft Predictive Maintenance System</h1><p>Setup is working!</p>'

if __name__ == '__main__':
    print("🚀 Test server starting...")
    print("📱 Open browser and go to: http://localhost:5000")
    app.run(debug=True, port=5000)