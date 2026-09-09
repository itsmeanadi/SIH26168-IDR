package com.idr.app;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.os.Bundle;
import android.webkit.WebView;
import org.json.JSONObject;
import org.json.JSONException;

public class SensorBridge implements SensorEventListener, LocationListener {
    private final WebView webView;
    private final SensorManager sensorManager;
    private final LocationManager locationManager;

    private float accX, accY, accZ;
    private float gyroX, gyroY, gyroZ;
    private float magX, magY, magZ;
    private double lat, lon, alt;
    private float accuracy;

    public SensorBridge(Context context, WebView webView) {
        this.webView = webView;
        this.sensorManager = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
        this.locationManager = (LocationManager) context.getSystemService(Context.LOCATION_SERVICE);

        setupSensors();
    }

    private void setupSensors() {
        Sensor acc = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        Sensor gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        Sensor mag = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);

        if (acc != null) sensorManager.registerListener(this, acc, SensorManager.SENSOR_DELAY_GAME);
        if (gyro != null) sensorManager.registerListener(this, gyro, SensorManager.SENSOR_DELAY_GAME);
        if (mag != null) sensorManager.registerListener(this, mag, SensorManager.SENSOR_DELAY_GAME);
    }

    public void startLocationUpdates() {
        try {
            locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 1000, 1.0f, this);
        } catch (SecurityException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            accX = event.values[0];
            accY = event.values[1];
            accZ = event.values[2];
        } else if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            gyroX = event.values[0];
            gyroY = event.values[1];
            gyroZ = event.values[2];
        } else if (event.sensor.getType() == Sensor.TYPE_MAGNETIC_FIELD) {
            magX = event.values[0];
            magY = event.values[1];
            magZ = event.values[2];
        }

        // We don't push on every sensor change to avoid flooding the JS bridge.
        // The MainActivity will trigger a periodic sync or we can do it here sparingly.
    }

    public void syncToJs() {
        try {
            JSONObject data = new JSONObject();
            JSONObject imu = new JSONObject();
            imu.put("acc_x", accX);
            imu.put("acc_y", accY);
            imu.put("acc_z", accZ);
            imu.put("gyro_x", gyroX);
            imu.put("gyro_y", gyroY);
            imu.put("gyro_z", gyroZ);
            imu.put("mag_x", magX);
            imu.put("mag_y", magY);
            imu.put("mag_z", magZ);

            JSONObject gps = new JSONObject();
            gps.put("lat", lat);
            gps.put("lon", lon);
            gps.put("alt", alt);
            gps.put("accuracy", accuracy);

            data.put("imu", imu);
            data.put("gps", gps);
            data.put("timestamp", System.currentTimeMillis());

            final String json = data.toString();
            webView.post(() -> webView.evaluateJavascript("if(window.onAndroidSensorUpdate) { window.onAndroidSensorUpdate(" + json + "); }", null));
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {}

    @Override
    public void onLocationChanged(Location location) {
        lat = location.getLatitude();
        lon = location.getLongitude();
        alt = location.getAltitude();
        accuracy = location.getAccuracy();
    }

    @Override
    public void onStatusChanged(String provider, int status, Bundle extras) {}
}
