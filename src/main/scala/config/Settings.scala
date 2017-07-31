package config

import com.typesafe.config.ConfigFactory

/**
  * Created by sbellary on 5/21/2017.
  */
object Settings {
  private val config = ConfigFactory.load()

  object WebLogGen {
    private val webLogGen = config.getConfig("clickstream")
  }

  object XxX {

  }
}
