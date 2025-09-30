package com.example.bmws_app;

import android.content.Context;
import android.util.Log;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

// FileManager manages CRUD for files on app-specific storage.
public class FileManager {

    // Create file if it does not exist
    static void createFile(Context appContext, String filename){
        File file = new File(appContext.getFilesDir(), filename);
        boolean fileCreated = false;
        try {
            fileCreated = file.createNewFile();
        } catch (IOException e){
            Log.e("FileManager", String.format("Error when creating file '%s': %s", filename, e.getMessage()));
        }

        if(fileCreated){
            Log.i("FileManager", String.format("Created file '%s' on '%s'", filename, appContext.getFilesDir().toString()));
        }else{
            Log.i("FileManager", String.format("File '%s' already exists on '%s'", filename, appContext.getFilesDir().toString()));
        }
    }

    static void writeFile(Context appContext, String filename, String content){
        try(FileOutputStream fos = appContext.openFileOutput(filename, Context.MODE_PRIVATE | Context.MODE_APPEND)){
            fos.write((content + '\n').getBytes(StandardCharsets.UTF_8));
        } catch (IOException e) {
            Log.e("FileManager", String.format("Error when writing to file '%s': %s", filename, e.getMessage()));
        }
    }

    // Print contents of file
    static String readFile(Context appContext, String filename){
        String contents;
        FileInputStream fis = null;
        try {
            fis = appContext.openFileInput(filename);
        }catch (FileNotFoundException e){
            Log.e("FileManager", String.format("File '%s' could not be found to read: %s", filename, e.getMessage()));
        }
        InputStreamReader streamReader = new InputStreamReader(fis, StandardCharsets.UTF_8);
        StringBuilder builder = new StringBuilder();
        try(BufferedReader reader = new BufferedReader(streamReader)){
            String line = reader.readLine();
            while (line != null){
                builder.append(line).append('\n');
                line = reader.readLine();
            }
        }catch (IOException e){
            Log.e("FileManager", String.format("Error when reading file '%s': %s", filename, e.getMessage()));
        }finally {
            contents = builder.toString();
        }
        return contents;
    }

    static void deleteFile(Context appContext, String filename){
        File file = new File(appContext.getFilesDir(), filename);
        if(file.delete()){
            Log.i("FileManager", String.format("File '%s' deleted", filename));
        }else{
            Log.i("FileManager", String.format("Error when deleting file '%s'", filename));
        }
    }

    // Print all filenames on app-specific storage
    static void printFileList(Context appContext){
        File dir = appContext.getFilesDir();
        String[] files = appContext.fileList();
        Log.i("FileManager", String.format("Files on '%s'", dir));
        for (String name: files) {
            Log.i("FileManager", name);
        }
    }
}
