from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('driver_availability_dataset.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_availability', methods=['POST'])
def check_availability():
    data = request.json
    try:
        distance = float(data['distance'])
        hours_logged = float(data['hours_logged'])
        climate = int(data['climate'])
        traffic = int(data['traffic'])

        # Filter the dataframe based on the inputs
        available_drivers = df[(df['distance_driven'] >= distance) & 
                               (df['hours_logged'] >= hours_logged) &
                               (df['weather_condition'] == climate) &
                               (df['available'] == 1) &
                               (df['traffic_condition'] <= traffic)]
        
        num_available = len(available_drivers)
        return jsonify({'num_available': num_available})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
