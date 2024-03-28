package com.example.bmws_app;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;

public class MathTools {
    static double min(Double[] array) {
        double minVal = array[0];
        for (Double aDouble : array) {
            if (aDouble < minVal) {
                minVal = aDouble;
            }
        }
        return minVal;
    }

    // Replacement method for np.sign
    static Integer[] sign(Float[] array) {
        Integer[] outputArray = new Integer[array.length];
        for (int i = 0; i < array.length; i++) {
            if (array[i] > 0) {
                outputArray[i] = 1;
            } else if (array[i] == 0) {
                outputArray[i] = 0;
            } else {
                outputArray[i] = -1;
            }
        }
        return outputArray;
    }

    // Replacement method for np.sign
    static Integer[] sign(Double[] array) {
        Integer[] outputArray = new Integer[array.length];
        for (int i = 0; i < array.length; i++) {
            if (array[i] > 0) {
                outputArray[i] = 1;
            } else if (array[i] == 0) {
                outputArray[i] = 0;
            } else {
                outputArray[i] = -1;
            }
        }
        return outputArray;
    }

    // This method was implemented using integers because its usage is for a sign vector and that is an array of integers
    static Integer[] diff(Integer[] array) {
        int n = array.length - 1;
        Integer[] output = new Integer[n];
        for (int i = 0; i < n; i++) {
            int d = array[i + 1] - array[i];
            output[i] = d;
        }
        return output;
    }

    // Returns an array of indexes where array[i] == target
    static Integer[] whereEqual(Integer[] array, int target){
        List<Integer> outputList = new ArrayList<Integer>();
        for (int i = 0; i < array.length; i++){
            if(array[i] == target){
                outputList.add(i);
            }
        }
        return outputList.toArray(new Integer[0]);
    }

    // Overload method for boolean arrays
    static Integer[] whereEqual(Boolean[] array, boolean target){
        List<Integer> outputList = new ArrayList<Integer>();
        for (int i = 0; i < array.length; i++) {
            if(array[i] == target){
                outputList.add(i);
            }
        }
        return outputList.toArray(new Integer[0]);
    }

    // Get the desired elements (selecting/slicing these elements) from a source array containing the desired indexes
    static Integer[] getElements(Integer[] source, Integer[] indexes){
        Integer[] output = new Integer[indexes.length];
        int count = 0;
        for (int index: indexes) {
            output[count] = source[index];
            count++;
        }
        return output;
    }

    // Overload method for Float arrays
    static Float[] getElements(Float[] source, Integer[] indexes){
        Float[] output = new Float[indexes.length];
        int count = 0;
        for (int index: indexes) {
            output[count] = source[index];
            count++;
        }
        return output;
    }

    // Overload method for double arrays
    static double[] getElements(double[] source, Integer[] indexes){
        double[] output = new double[indexes.length];
        int count = 0;
        for (int index: indexes) {
            output[count] = source[index];
            count++;
        }
        return output;
    }

    // Returns an array containing all the indexes where array[i] > target
    static Integer[] whereGreater(Float[] array, float target){
        List<Integer> outputList = new ArrayList<Integer>();
        for (int i = 0; i < array.length; i++) {
            if(array[i] > target){
                outputList.add(i);
            }
        }
        return outputList.toArray(new Integer[0]);
    }

    // Overload method for double array
    static Integer[] whereGreater(double[] array, float target){
        List<Integer> outputList = new ArrayList<Integer>();
        for (int i = 0; i < array.length; i++) {
            if(array[i] > target){
                outputList.add(i);
            }
        }
        return outputList.toArray(new Integer[0]);
    }

    // Returns an array containing all the indexes where array[i] >= target
    static Integer[] whereGreaterEqual(Float[] array, float target){
        List<Integer> outputList = new ArrayList<Integer>();
        for (int i = 0; i < array.length; i++) {
            if(array[i] >= target){
                outputList.add(i);
            }
        }
        return outputList.toArray(new Integer[0]);
    }

    // Overload method for double array
    static Integer[] whereGreaterEqual(double[] array, float target){
        List<Integer> outputList = new ArrayList<Integer>();
        for (int i = 0; i < array.length; i++) {
            if(array[i] >= target){
                outputList.add(i);
            }
        }
        return outputList.toArray(new Integer[0]);
    }

    // Returns an array containing all the indexes where array[i] < target
    static Integer[] whereLess(Float[] array, float target){
        List<Integer> outputList = new ArrayList<Integer>();
        for (int i = 0; i < array.length; i++) {
            if(array[i] < target){
                outputList.add(i);
            }
        }
        return outputList.toArray(new Integer[0]);
    }

    // Overload method for double array
    static Integer[] whereLess(double[] array, float target){
        List<Integer> outputList = new ArrayList<Integer>();
        for (int i = 0; i < array.length; i++) {
            if(array[i] < target){
                outputList.add(i);
            }
        }
        return outputList.toArray(new Integer[0]);
    }

    // Returns an array containing all the indexes where array[i] <= target
    static Integer[] whereLessEqual(Float[] array, float target){
        List<Integer> outputList = new ArrayList<Integer>();
        for (int i = 0; i < array.length; i++) {
            if(array[i] <= target){
                outputList.add(i);
            }
        }
        return outputList.toArray(new Integer[0]);
    }

    // Overload method for double array
    static Integer[] whereLessEqual(double[] array, float target){
        List<Integer> outputList = new ArrayList<Integer>();
        for (int i = 0; i < array.length; i++) {
            if(array[i] <= target){
                outputList.add(i);
            }
        }
        return outputList.toArray(new Integer[0]);
    }

    // Returns the index of the lowest value in the array
    static int argmin(Double[] array){
        double minValue = array[0];
        int minIndex = 0;
        for (int i = 0; i < array.length; i++) {
            if(array[i] < minValue){
                minValue = array[i];
                minIndex = i;
            }
        }
        return minIndex;
    }

    static float mean(Float[] array){
        float sum = 0;
        for (float f: array) {
            sum += f;
        }
        return sum/array.length;
    }

    // Overload method for double array
    static float mean(double[] array){
        float sum = 0;
        for (double d: array) {
            sum += d;
        }
        return sum/array.length;
    }

    // Generates an array of false
    static Boolean[] ones(int columns){
        Boolean[] output = new Boolean[columns];
        for (int i = 0; i < columns; i++) {
            output[i] = true;
        }
        return output;
    }

    // Sums a boolean array (true = 1, false = 0)
    static int boolSum(Boolean[] boolArray){
        int count = 0;
        for (Boolean value: boolArray) {
            if(value){
                count++;
            }
        }
        return count;
    }

    // Overload method for boolean list
    static int boolSum(List<Boolean> boolList){
        int count = 0;
        for (Boolean value: boolList){
            if(value){
                count++;
            }
        }
        return count;
    }

    // Returns a boolean array where each element is true or false depending whether listA[i] is on listB[i]
    static Boolean[] isIn(List<Integer> listA, List<Integer> listB){
        Boolean[] output = new Boolean[listA.size()];
        for (int i = 0; i < listA.size(); i++) {
            output[i] = listB.contains(listA.get(i));
        }
        return output;
    }

    // Utility method
    static double[] toDoubleArray(Float[] array){
        double[] output = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            output[i] = array[i];
        }
        return output;
    }

    // Utility method
    static double[] toDoubleArray(Double[] array){
        double[] output = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            output[i] = array[i];
        }
        return output;
    }

    // Utility method
    static Double[] toDoubleArray(double[] array){
        Double[] output = new Double[array.length];
        for (int i = 0; i < array.length; i++) {
            output[i] = array[i];
        }
        return output;
    }

    // Utility method
    static Double[] toDoubleArray(ArrayDeque<Integer> arrayDeque){
        Integer[] outputInteger = arrayDeque.toArray(new Integer[0]);
        Double[] outputDouble = new Double[arrayDeque.size()];
        for (int i = 0; i < arrayDeque.size(); i++){
            outputDouble[i] = Double.valueOf(outputInteger[i]);
        }
        return outputDouble;
    }

    // Utility method
    static Float[] toFloatArray(Double[] array){
        Float[] output = new Float[array.length];
        for (int i = 0; i < array.length; i++) {
            output[i] = array[i].floatValue();
        }
        return output;
    }

    // Gets the last n elements from array
    static Double[] extractArray(Double[] array, int n){
        // DEBUG ------------
        //Log.i("ExtractArray", String.format("Array Length: %s, n: %s", array.length, n));
        // ------------------
        Double[] output = new Double[n];
        int outIndex = 0;
        for (int i = array.length - n; i < array.length; i++){
            output[outIndex] = array[i];
            outIndex++;
        }
        return output;
    }
}
