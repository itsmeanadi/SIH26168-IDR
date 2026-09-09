package com.idr.app;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Handler;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;

public class MainActivity extends Activity {
    private WebView webView;
    private SensorBridge sensorBridge;
    private Handler syncHandler = new Handler();

    private final String LOCAL_URL = "http://10.89.225.17:8000";
    private final String RENDER_URL = "https://sih26168-idr-3.onrender.com";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        webView = findViewById(R.id.webView);
        configureWebView();

        requestPermissions();

        sensorBridge = new SensorBridge(this, webView);

        // Start periodic sync to JS at 10Hz
        syncHandler.post(syncRunnable);

        // Determine backend to load
        String finalUrl = determineBackend();
        webView.loadUrl(finalUrl);
    }

    private String determineBackend() {
        if (isBackendAvailable(LOCAL_URL)) {
            return LOCAL_URL;
        } else if (isBackendAvailable(RENDER_URL)) {
            return RENDER_URL;
        }
        return RENDER_URL; // Default fallback
    }

    private boolean isBackendAvailable(String urlString) {
        try {
            URL url = new URL(urlString + "/api/system/health");
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setConnectTimeout(2000);
            connection.setReadTimeout(2000);
            connection.setRequestMethod("GET");
            int responseCode = connection.getResponseCode();
            return (responseCode == HttpURLConnection.HTTP_OK);
        } catch (IOException e) {
            return false;
        }
    }

    private void configureWebView() {
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setAllowFileAccess(true);
        settings.setAllowContentAccess(true);
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_ALWAYS_ALLOW);

        webView.setWebViewClient(new WebViewClient());
    }

    private void requestPermissions() {
        String[] permissions = {
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION
        };

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, permissions, 100);
        }
    }

    private final Runnable syncRunnable = new Runnable() {
        @Override
        public void run() {
            if (sensorBridge != null) {
                sensorBridge.syncToJs();
            }
            syncHandler.postDelayed(this, 100); // 10Hz
        }
    };

    @Override
    protected void onResume() {
        super.onResume();
        if (sensorBridge != null) {
            sensorBridge.startLocationUpdates();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
    }
}
