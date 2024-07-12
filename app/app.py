from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)


model = joblib.load('app/model/ModeloRegresionForest2.pkl')
app.logger.debug('Modelo cargado correctamente.')


@app.route("/", methods=["GET", "POST"]) 
def home():
    
    
    return render_template("model.html")


@app.route('/predict_model', methods=['POST'])
def predict():
    try:
        ordinal = OrdinalEncoder()
        # Obtener los datos enviados en el request
        carat = float(request.form['carat'])
        color = request.form['color']
        clarity = request.form['clarity']
        x = float(request.form['x'])
        y = float(request.form['y'])
        z = float(request.form['z'])

        
        columns = ['carat', 'color', 'clarity', 'x', 'y', 'z']
        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[carat, color, clarity, x, y, z]], columns=columns)
        
        # Preprocesar los datos
        obtFeactures = ['color', 'clarity']
        for feature in obtFeactures:
            data_df[feature] = ordinal.fit_transform(data_df[[feature]])
            app.logger.debug(f'Datos preprocesados: {data_df}')
        

        app.logger.debug(f'DataFrame creado: {data_df}')

        # Escalar los datosscaled = StandardScaler()
       
                
        # Realizar predicciones
        prediction = model.predict(data_df)
        app.logger.debug(f'Predicción: {prediction[0]}')
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'price': prediction[0]})
    
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)