def fullscreen():
    from pymouse import PyMouse
    from pykeyboard import PyKeyboard

    m = PyMouse()
    k = PyKeyboard()

    x_dim, y_dim = m.screen_size()
    m.click(int(x_dim/3), int(y_dim/2), 1)
    k.press_key(k.control_key)
    k.tap_key(k.function_keys[11])
    k.release_key(k.control_key)

