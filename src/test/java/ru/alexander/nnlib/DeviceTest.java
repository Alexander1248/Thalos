package ru.alexander.nnlib;

import com.aparapi.device.Device;
import com.aparapi.internal.kernel.KernelManager;
import com.aparapi.internal.kernel.KernelPreferences;
import junit.framework.TestCase;
import ru.alexander.nnlib.types.ActivationFunctionType;
import ru.alexander.nnlib.types.ThreadingType;

public class DeviceTest extends TestCase {
    public void testResult() {
        KernelPreferences preferences = KernelManager.instance().getDefaultPreferences();
        System.out.println("-- Devices in preferred order --");
        for (Device device : preferences.getPreferredDevices(null)) {
            System.out.println("----------");
            System.out.println(device);
        }
    }
}
