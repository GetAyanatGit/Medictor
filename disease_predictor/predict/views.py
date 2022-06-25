import time
from django.http import HttpResponseRedirect
from django.shortcuts import redirect, render
from matplotlib.style import context
from .forms import SymptomsForm
from collections import Counter
import pickle
import pandas as pd
import numpy as np
from django.contrib import messages

# Will be the resultant prediction
predicted_disease = ''

# We are loading our machine learning model dump here
# To know how we created dump refer to machine learning model in Kaggle

model = pickle.load(open(r"C:\Users\Ayan Mukherjee\Desktop\DISEASE_PREDICTOR\disease_predictor\predict\svc_ml_model.sav", "rb"))

# In order to process label encoded input for our machine learning model
# we need to know the encoding values for each input symptoms and process them to the model
df1 = pd.read_csv(r'C:\Users\Ayan Mukherjee\Desktop\DISEASE_PREDICTOR\disease_predictor\svcmodel\Symptom-severity.csv')
a = np.array(df1["Symptom"])
b = np.array(df1["weight"])


processed = [('itching', 'Itching'), ('skin_rash', 'Skin rash'), ('nodal_skin_eruptions', 'Nodal skin eruptions'), ('continuous_sneezing', 'Continuous sneezing'), ('shivering', 'Shivering'), ('chills', 'Chills'), ('joint_pain', 'Joint pain'), ('stomach_pain', 'Stomach pain'), ('acidity', 'Acidity'), ('ulcers_on_tongue', 'Ulcers on tongue'), ('muscle_wasting', 'Muscle wasting'), ('vomiting', 'Vomiting'), ('burning_micturition', 'Burning micturition'), ('spotting_urination', 'Spotting urination'), ('fatigue', 'Fatigue'), ('weight_gain', 'Weight gain'), ('anxiety', 'Anxiety'), ('cold_hands_and_feets', 'Cold hands and feets'), ('mood_swings', 'Mood swings'), ('weight_loss', 'Weight loss'), ('restlessness', 'Restlessness'), ('lethargy', 'Lethargy'), ('patches_in_throat', 'Patches in throat'), ('irregular_sugar_level', 'Irregular sugar level'), ('cough', 'Cough'), ('high_fever', 'High fever'), ('sunken_eyes', 'Sunken eyes'), ('breathlessness', 'Breathlessness'), ('sweating', 'Sweating'), ('dehydration', 'Dehydration'), ('indigestion', 'Indigestion'), ('headache', 'Headache'), ('yellowish_skin', 'Yellowish skin'), ('dark_urine', 'Dark urine'), ('nausea', 'Nausea'), ('loss_of_appetite', 'Loss of appetite'), ('pain_behind_the_eyes', 'Pain behind the eyes'), ('back_pain', 'Back pain'), ('constipation', 'Constipation'), ('abdominal_pain', 'Abdominal pain'), ('diarrhoea', 'Diarrhoea'), ('mild_fever', 'Mild fever'), ('yellow_urine', 'Yellow urine'), ('yellowing_of_eyes', 'Yellowing of eyes'), ('acute_liver_failure', 'Acute liver failure'), ('fluid_overload', 'Fluid overload'), ('swelling_of_stomach', 'Swelling of stomach'), ('swelled_lymph_nodes', 'Swelled lymph nodes'), ('malaise', 'Malaise'), ('blurred_and_distorted_vision', 'Blurred and distorted vision'), ('phlegm', 'Phlegm'), ('throat_irritation', 'Throat irritation'), ('redness_of_eyes', 'Redness of eyes'), ('sinus_pressure', 'Sinus pressure'), ('runny_nose', 'Runny nose'), ('congestion', 'Congestion'), ('chest_pain', 'Chest pain'), ('weakness_in_limbs', 'Weakness in limbs'), ('fast_heart_rate', 'Fast heart rate'), ('pain_during_bowel_movements', 'Pain during bowel movements'), ('pain_in_anal_region', 'Pain in anal region'), ('bloody_stool', 'Bloody stool'), ('irritation_in_anus', 'Irritation in anus'), ('neck_pain', 'Neck pain'), ('dizziness', 'Dizziness'), ('cramps', 'Cramps'), ('bruising', 'Bruising'), ('obesity', 'Obesity'), ('swollen_legs', 'Swollen legs'), ('swollen_blood_vessels', 'Swollen blood vessels'), ('puffy_face_and_eyes', 'Puffy face and eyes'), ('enlarged_thyroid', 'Enlarged thyroid'), ('brittle_nails', 'Brittle nails'), ('swollen_extremeties', 'Swollen extremeties'), ('excessive_hunger', 'Excessive hunger'), ('extra_marital_contacts', 'Extra marital contacts'), ('drying_and_tingling_lips', 'Drying and tingling lips'), ('slurred_speech', 'Slurred speech'), ('knee_pain', 'Knee pain'), ('hip_joint_pain', 'Hip joint pain'), ('muscle_weakness', 'Muscle weakness'), ('stiff_neck', 'Stiff neck'), ('swelling_joints', 'Swelling joints'), ('movement_stiffness', 'Movement stiffness'), ('spinning_movements', 'Spinning movements'), ('loss_of_balance', 'Loss of balance'), ('unsteadiness', 'Unsteadiness'), ('weakness_of_one_body_side', 'Weakness of one body side'), ('loss_of_smell', 'Loss of smell'), ('bladder_discomfort', 'Bladder discomfort'), ('foul_smell_ofurine', 'Foul smell ofurine'), ('continuous_feel_of_urine', 'Continuous feel of urine'), ('passage_of_gases', 'Passage of gases'), ('internal_itching', 'Internal itching'), ('toxic_look_(typhos)', 'Toxic look (typhos)'), ('depression', 'Depression'), ('irritability', 'Irritability'), ('muscle_pain', 'Muscle pain'), ('altered_sensorium', 'Altered sensorium'), ('red_spots_over_body', 'Red spots over body'), ('belly_pain', 'Belly pain'), ('abnormal_menstruation', 'Abnormal menstruation'), ('dischromic_patches', 'Dischromic patches'), ('watering_from_eyes', 'Watering from eyes'), ('increased_appetite', 'Increased appetite'), ('polyuria', 'Polyuria'), ('family_history', 'Family history'), ('mucoid_sputum', 'Mucoid sputum'), ('rusty_sputum', 'Rusty sputum'), ('lack_of_concentration', 'Lack of concentration'), ('visual_disturbances', 'Visual disturbances'), ('receiving_blood_transfusion', 'Receiving blood transfusion'), ('receiving_unsterile_injections', 'Receiving unsterile injections'), ('coma', 'Coma'), ('stomach_bleeding', 'Stomach bleeding'), ('distention_of_abdomen', 'Distention of abdomen'), ('history_of_alcohol_consumption', 'History of alcohol consumption'), ('blood_in_sputum', 'Blood in sputum'), ('prominent_veins_on_calf', 'Prominent veins on calf'), ('palpitations', 'Palpitations'), ('painful_walking', 'Painful walking'), ('pus_filled_pimples', 'Pus filled pimples'), ('blackheads', 'Blackheads'), ('scurring', 'Scurring'), ('skin_peeling', 'Skin peeling'), ('silver_like_dusting', 'Silver like dusting'), ('small_dents_in_nails', 'Small dents in nails'), ('inflammatory_nails', 'Inflammatory nails'), ('blister', 'Blister'), ('red_sore_around_nose', 'Red sore around nose'), ('yellow_crust_ooze', 'Yellow crust ooze'), ('prognosis', 'Prognosis')]



# This is still now an one-page application, thus only has one view

def get_name(request):
    empty_flag = '' # As of now no inputs, doesn't matter if all inputs are empty

    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        request.method = 'None'
        empty_flag = ''
        print('Posting')
        #empty_flag = ''
        # create a form instance and populate it with data from the request:
        form = SymptomsForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            data = form.cleaned_data

            # Cleaned data is a dictionary hence we have to fetch the values in an array
            symptom_lists = list(data.keys())
            symptoms = []
            for i in symptom_lists:
                symptoms.append(data[i])

            # Now hashing with the keys we have taken the symptoms in a list
            
            # Setting up a counter to track input deplicaices
            c = Counter(symptoms)

            # Pro-actively encoding the value of 'None' in symptoms to 0 since it will have no weightage
            for i in range(0,len(symptoms)):
                if symptoms[i] == 'None':
                    symptoms[i] = 0

            # Note that we have made a counter but 
            if c['None']==5:
                print('Entering')
                print('At least one entry required') # error msg
                empty_flag = 'all_empty' # Empty flag is set to 0
               
            
            else:
                print(symptoms)
                # Encoding Symptoms numerical values for ML input
                for j in range(len(symptoms)):
                    for k in range(len(a)):
                        if symptoms[j]==a[k]:
                            symptoms[j]=b[k]
                        
                # Refer to ML model in Kaggle, we have decided to consider the 1st 5 features only
                # Hence weightage of the rest 13 is set to 0
                nulls = [0,0,0,0,0,0,0,0,0,0,0,0]
                
                # Input features to be predicted against
                psy = [symptoms + nulls]

                # Assigning the returned answer to predicted variable
                predicted = model.predict(psy)
                print(predicted[0])
                predicted_disease = predicted[0]


            if empty_flag == 'all_empty':
                pass
                # Simply pass away
            else:
                #  This means user posted with some symptoms
                #  Send the result as a message to the front end

                message = 'Patient has symptoms of ' + predicted_disease
                return render(request, 'predict/formpage.html', {'form': form,'messages': message})
            
    else:
        # If there is no POST request yet, then we just need to initialize blank form
        print('else part')
        form = SymptomsForm()

    if empty_flag == 'all_empty':
        # As per this flag alert the user
        return render(request, 'predict/formpage.html', {'form': form,'messages': 'At least One Input Required !!'})
    else:
        # Else if there is no request simply render the blank form
        return render(request, 'predict/formpage.html', {'form': form})



