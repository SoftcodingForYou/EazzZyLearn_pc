package com.example.bmws_app;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

public class JsonParser{
    static JSONObject parse(String jsonString){
        JSONObject object = null;
        try{
            object = new JSONObject(jsonString);
        } catch(JSONException e){
            e.printStackTrace();
        }
        return object;
    }

    static double[] toDoubleArray(JSONArray array) throws JSONException {
        double[] output = new double[array.length()];
        for(int i = 0; i < array.length(); i++){
            output[i] = array.getDouble(i);
        }
        return output;
    }
}
