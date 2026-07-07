from cirque import servicehubutils
from cirque import pimc
from cirque import servicehubcontrol


def initialize_platform(
    platform_ip: str, verbose: bool = False
) -> servicehubutils.FPGAConnection:
    # Define connection (won't fail if device is not available)
    con = servicehubutils.FPGAConnection(ip=platform_ip, port=50058)

    if verbose:
        print(f"Connection is open: {con.is_open()}")
        print(f"Channel {con.get_channel()}")

        # Connect to servicehub and to the PIMC
        my_pimc = pimc.PIMC(con)
        print(f"Platform ready: {my_pimc.get_platform_ready()}")

    return con


def print_endpoints(con: servicehubutils.FPGAConnection):
    my_servicehubcontrol = servicehubcontrol.ServicehubControl(con)
    # Get all ServiceHub Plugins
    hub_modules = my_servicehubcontrol.get_plugin_list()
    print("Available endpoints:")
    for module in hub_modules:
        print(f"{module}: {my_servicehubcontrol.get_endpoints_of_plugin(module)}")
