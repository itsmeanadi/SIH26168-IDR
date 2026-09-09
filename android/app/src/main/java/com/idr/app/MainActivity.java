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

public class MainActivity extends Activity {
    private WebView webView;
    private SensorBridge sensorBridge;
    private Handler syncHandler = new Handler();
    private final String SERVER_URL = "http://10.89.225.17:8000"; // User should update this to their server IP

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

        webView.loadUrl(SERVER_URL);
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
        // Stop sensor updates if needed to save battery
    }
}
