from flask import Flask,request,render_template,jsonify
from src.pipelines.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict' , methods = ['GET' , 'POST'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template('form.html')
    
    else:
        # making object of custom data class
        data = CustomData(
            meanRadius = float(request.form.get('meanRadius')),
            meanTexture = float(request.form.get('meanTexture')),
            meanPerimeter = float(request.form.get('meanPerimeter')),
            meanArea = float(request.form.get('meanArea')),
            meanSmoothness = float(request.form.get('meanSmoothness')),
            meanCompactness = float(request.form.get('meanCompactness')),
            meanConcavity = float(request.form.get('meanConcavity')),
            meanConcavePoints = float(request.form.get('meanConcavePoints')),
            meanSymmetry = float(request.form.get('meanSymmetry')),
            meanFractalDimension = float(request.form.get('meanFractalDimension')),
            radiusError = float(request.form.get('radiusError')),
            textureError = float(request.form.get('textureError')),
            perimeterError = float(request.form.get('perimeterError')),
            areaError = float(request.form.get('areaError')),
            smoothnessError = float(request.form.get('smoothnessError')),
            compactnessError = float(request.form.get('compactnessError')),
            concavityError = float(request.form.get('concavityError')),
            concavePointsError = float(request.form.get('concavePointsError')),
            symmetryError = float(request.form.get('symmetryError')),
            fractalDimensionError = float(request.form.get('fractalDimensionError')),
            worstRadius = float(request.form.get('worstRadius')),
            worstTexture = float(request.form.get('worstTexture')),
            worstPerimeter = float(request.form.get('worstPerimeter')),
            worstArea = float(request.form.get('worstArea')),
            worstSmoothness = float(request.form.get('worstSmoothness')),
            worstCompactness = float(request.form.get('worstCompactness')),
            worstConcavity = float(request.form.get('worstConcavity')),
            worstConcavePoints = float(request.form.get('worstConcavePoints')),
            worstSymmetry = float(request.form.get('worstSymmetry')),
            worstFractalDimension = float(request.form.get('worstFractalDimension'))

        )

        # making the dataframe by calling obj method
        final_new_data = data.get_data_as_dataframe()

        # making object of predictPipeline class
        predict_pipeline = PredictPipeline()

        pred = predict_pipeline.predict(final_new_data)


        if int(pred[0]) == 1:
            results = 'malignant(cancerous)'

        else:
            results = 'benign(non-cancerous)'

        return render_template('form.html' , final_result = results )
    

if __name__ == "__main__":
    app.run(host = '0.0.0.0' , debug = True)


    
